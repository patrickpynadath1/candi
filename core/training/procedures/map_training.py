"""Procedure adapters for continuous maps; numerical losses stay composable."""

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from .maps import LinearInterpolation, iterate_maps
from .losses import TokenCrossEntropy, MaskedMean


@contextmanager
def deterministic(model):
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)


class MapTrainingStates(nn.Module):
    def __init__(self, interpolation=None, warmup_steps=100):
        super().__init__()
        self.interpolation = interpolation or LinearInterpolation()
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")

    def forward(self, models, batch, step):
        tokens = batch["input_ids"]
        source = torch.rand(len(tokens), device=tokens.device) * 0.999 + 0.001
        destination = source * torch.rand_like(source) * 0.999
        middle = (source + destination) / 2
        corruption = batch.get("corruption_mask", batch["loss_mask"]).bool()
        vocab = models["student"].model.vocab_size
        state, velocity = self.interpolation.sample(
            tokens, vocab, source, mask=corruption
        )
        return dict(
            state=state,
            velocity=velocity,
            source=source,
            middle=middle,
            destination=destination,
            tokens=tokens,
            clean=F.one_hot(tokens, vocab).float(),
            corruption_mask=corruption,
            mask=batch["loss_mask"].bool(),
            consistency_enabled=step >= self.warmup_steps,
        )


class MapTargets(nn.Module):
    def __init__(self, consistency_target, endpoint=False):
        super().__init__()
        self.consistency_target = consistency_target
        self.endpoint = bool(endpoint)

    def forward(self, models, state):
        model = models["student"].model
        target = None
        if state["consistency_enabled"]:
            with deterministic(model):
                target = (
                    self.consistency_target(model, state["state"], state["velocity"])
                    if self.endpoint
                    else self.consistency_target(
                        model,
                        state["state"],
                        state["source"],
                        state["middle"],
                        state["destination"],
                        prompt_state=state["clean"],
                        prompt_mask=~state["corruption_mask"],
                    )
                )
        return dict(diagonal=state["tokens"], consistency=target)


class MapPredictions(nn.Module):
    def __init__(self, endpoint=False):
        super().__init__()
        self.endpoint = bool(endpoint)

    def forward(self, models, state):
        model = models["student"].model
        with deterministic(model):
            # Autonomous maps use the clean boundary condition, not noisy CE.
            diagonal = (
                model(state["clean"])
                if self.endpoint
                else model(state["state"], state["source"], state["source"])
            )
            consistency = None
            if state["consistency_enabled"]:
                consistency = (
                    model(state["state"]).exp()
                    if self.endpoint
                    else model(state["state"], state["source"], state["destination"])
                )
        return dict(diagonal=diagonal, consistency=consistency)


class ForwardKL(nn.Module):
    def forward(self, prediction, target, mask):
        # prediction is log probabilities; target is probabilities.
        log_target = target.clamp_min(torch.finfo(target.dtype).tiny).log()
        return (target * (log_target - prediction)).sum(-1)


class MapLoss(nn.Module):
    """Boundary/diagonal CE plus an independently configured consistency loss."""

    def __init__(self, consistency, coefficient=1.0):
        super().__init__()
        if not 0 <= coefficient < float("inf"):
            raise ValueError("coefficient must be nonnegative and finite")
        self.diagonal = TokenCrossEntropy()
        self.consistency = consistency
        self.coefficient = float(coefficient)

    def forward(self, prediction, target, mask):
        loss = self.diagonal(prediction["diagonal"], target["diagonal"], mask)
        if target["consistency"] is not None:
            loss = loss + self.coefficient * self.consistency(
                prediction["consistency"], target["consistency"], mask
            )
        return loss


class MapAlgorithm(nn.Module):
    """Standard training/evaluation surface for a configurable map predictor."""

    validation_metric = "val/diagonal_ce"

    def __init__(self, model, interpolation=None, length=180, endpoint=False):
        super().__init__()
        self.model = model
        self.interpolation = interpolation or LinearInterpolation()
        self.length = int(length)
        self.endpoint = bool(endpoint)

    def loss(self, tokens, loss_mask, corruption_mask=None, **kwargs):
        time = torch.rand(len(tokens), device=tokens.device).clamp_min(1e-5)
        mask = loss_mask.bool() if corruption_mask is None else corruption_mask.bool()
        state, _ = self.interpolation.sample(
            tokens, self.model.vocab_size, time, mask=mask
        )
        if self.endpoint:
            state = F.one_hot(tokens, self.model.vocab_size).float()
        prediction = (
            self.model(state) if self.endpoint else self.model(state, time, time)
        )
        values = TokenCrossEntropy()(prediction, tokens, loss_mask)
        loss, total, count = MaskedMean()(values, loss_mask)
        return dict(
            loss=loss, loss_sum=total, num_tokens=count, normalization_count=count
        )

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        with deterministic(self):
            return sampler.sample(self, prompt_tokens, prompt_mask)


class MapSampler:
    """One map/refinement executor with a selected step policy."""

    def __init__(self, step, steps=1):
        if int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        self.step, self.steps = step, int(steps)

    def steps_for_nfe(self, nfe):
        if int(nfe) != nfe or nfe < 1:
            raise ValueError("NFE must be a positive integer")
        return int(nfe)

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        clean = F.one_hot(prompt_tokens, algo.model.vocab_size).float()
        state = torch.randn_like(clean) * algo.interpolation.noise_scale
        times = torch.linspace(1, 0, self.steps + 1, device=state.device)
        state, nfe = iterate_maps(
            algo.model, state, times, self.step, clean, prompt_mask
        )
        return state.argmax(-1), dict(nfe=nfe)
