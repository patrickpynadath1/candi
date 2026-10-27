"""Map scaffolding on an inherited affine-Gaussian information path.

PSD weights follow composition of z_s=A_ts*z_t+B_ts*E[x0]. The endpoint
variant enforces characteristic consistency in (state,time); it is a
schedule-aware Beckmann-style extension, not the autonomous paper reproduction.
"""

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from core.models.layers import TimestepEmbedder
from core.processes.gaussian import GaussianProcess
from .map_training import deterministic
from .losses import TokenCrossEntropy, MaskedMean


class AffineMap(nn.Module):
    validation_metric = "val/diagonal_ce"

    def __init__(self, base, endpoint=False):
        super().__init__()
        if len(base.views) != 1 or not isinstance(
            base.views[0].process, GaussianProcess
        ):
            raise ValueError("affine maps require one Gaussian view")
        if base.prediction.enabled:
            raise ValueError("affine maps currently require a no-SC base")
        if not base.model.backbone.adaLN:
            raise ValueError("destination conditioning requires an AdaLN backbone")
        representation = base.views[0].representation
        codebook = getattr(representation, "state_representation", representation)
        if any(p.requires_grad for p in codebook.parameters()):
            raise ValueError("affine distillation requires a fixed state codebook")
        self.base, self.endpoint = base, bool(endpoint)
        self.length = base.length
        if not endpoint:
            size = base.model.backbone.sigma_map.mlp[-1].out_features
            self.destination_time = TimestepEmbedder(size)
            nn.init.zeros_(self.destination_time.mlp[-1].weight)
            nn.init.zeros_(self.destination_time.mlp[-1].bias)

    @property
    def model(self):
        return self.base.model

    @property
    def initialization_target(self):
        return self.base

    @property
    def view(self):
        return self.base.views[0]

    def coefficients(self, time):
        return self.base.points(time)[0].coefficients

    def forward(self, state, source, destination):
        points = self.base.points(source)
        hidden, condition = self.base.model.prepare_fused((state,), points, source)
        if not self.endpoint:
            condition = condition + self.destination_time(destination.float())
        logits = self.base.model._slice_output(
            self.base.model.backbone.forward_hidden(hidden, condition)
        )
        # Preserve the base's posterior parameterization, including softcaps.
        return self.base.predictor.training_posterior(
            logits, self.base._observed((state,))
        )

    def factors(self, source, destination):
        t, s = self.coefficients(source), self.coefficients(destination)
        if torch.any(t.sigma <= 0):
            raise ValueError("map source must have positive noise")
        a = s.sigma / t.sigma
        b = s.alpha - a * t.alpha
        return a, b

    def transport(self, state, probabilities, source, destination):
        a, b = self.factors(source, destination)
        mean = self.view.representation.posterior_mean(probabilities)
        return a[:, None, None] * state + b[:, None, None] * mean

    def loss(self, tokens, loss_mask, corruption_mask=None, **kwargs):
        mask = loss_mask.bool() if corruption_mask is None else corruption_mask.bool()
        time = torch.rand(len(tokens), device=tokens.device) * 0.998 + 0.001
        clean = self.view.representation.encode(tokens)
        state = self.view.process.corrupt_state(clean, self.coefficients(time), mask)
        values = TokenCrossEntropy()(self(state, time, time), tokens, loss_mask)
        loss, total, count = MaskedMean()(values, loss_mask)
        return dict(
            loss=loss, loss_sum=total, normalization_count=count, num_tokens=count
        )

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        with deterministic(self):
            return sampler.sample(self, prompt_tokens, prompt_mask)


class AffineStates(nn.Module):
    def forward(self, models, batch, step):
        model = models["student"]
        tokens = batch["input_ids"]
        source = torch.rand(len(tokens), device=tokens.device) * 0.998 + 0.001
        destination = source * torch.rand_like(source) * 0.95
        middle = (source + destination) / 2
        mask = batch.get("corruption_mask", batch["loss_mask"]).bool()
        clean = model.view.representation.encode(tokens)
        state = model.view.process.corrupt_state(
            clean, model.coefficients(source), mask
        )
        return dict(
            state=state,
            clean=clean,
            source=source,
            middle=middle,
            destination=destination,
            corruption_mask=mask,
            tokens=tokens,
            mask=batch["loss_mask"].bool(),
        )


class AffinePredictions(nn.Module):
    def forward(self, models, state):
        model = models["student"]
        with deterministic(model):
            if model.endpoint:
                zero = torch.zeros_like(state["source"])
                diagonal = model(state["clean"], zero, zero)
                consistency = model(state["state"], state["source"], zero).exp()
            else:
                diagonal = model(state["state"], state["source"], state["source"])
                consistency = model(
                    state["state"], state["source"], state["destination"]
                )
        return dict(diagonal=diagonal, consistency=consistency)


class AffinePSDTarget(nn.Module):
    @torch.no_grad()
    def forward(self, models, state):
        teacher = models["teacher"]
        t, u, s = (state[k] for k in ("source", "middle", "destination"))
        with deterministic(teacher):
            first = teacher(state["state"], t, u).exp()
            intermediate = teacher.transport(state["state"], first, t, u)
            intermediate = torch.where(
                state["corruption_mask"][..., None], intermediate, state["clean"]
            )
            second = teacher(intermediate, u, s).exp()
            _, b_tu = teacher.factors(t, u)
            a_us, b_us = teacher.factors(u, s)
            _, b_ts = teacher.factors(t, s)
            # Flat adaptive schedule segments contain no denoising information.
            active = b_ts.abs() > 1e-7
            weight = torch.where(active, a_us * b_tu / b_ts.clamp_min(1e-7), 1.0)
            if torch.any((weight < -1e-4) | (weight > 1.0001)) or torch.any(
                b_us < -1e-6
            ):
                raise ValueError("PSD needs monotone affine signal/noise ratios")
            weight = weight.clamp(0, 1)[:, None, None]
            target = weight * first + (1 - weight) * second
        return dict(diagonal=state["tokens"], consistency=target)


class CharacteristicTarget(nn.Module):
    """Detached endpoint + directional derivative along the teacher ODE.

    Include the time derivative: an adaptive schedule's clock is part of the
    function. Coefficient slopes are finite differences of the actual inherited
    path, including any adaptive time map, rather than bare local derivatives.
    """

    def __init__(self, difference_step=1e-3):
        super().__init__()
        self.difference_step = difference_step

    def forward(self, models, state):
        teacher, student = models["teacher"], models["student"]
        x, t = state["state"], state["source"]
        with torch.no_grad(), deterministic(teacher):
            posterior = teacher(x, t, torch.zeros_like(t)).exp()
            mean = teacher.view.representation.posterior_mean(posterior)
            lo, hi = (
                (t - self.difference_step).clamp_min(0),
                (t + self.difference_step).clamp_max(1),
            )
            a, b, c = (
                teacher.coefficients(lo),
                teacher.coefficients(hi),
                teacher.coefficients(t),
            )
            da = (b.alpha - a.alpha) / (hi - lo)
            ds = (b.sigma - a.sigma) / (hi - lo)
            velocity = teacher.view.process.reverse_vector_field(
                x, mean, c.alpha, c.sigma, da, ds
            )
            velocity = torch.where(state["corruption_mask"][..., None], velocity, 0.0)
        with deterministic(student), torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            value, tangent = torch.autograd.functional.jvp(
                lambda z, time: student(z, time, torch.zeros_like(time)).exp(),
                (x, t),
                (-velocity, -torch.ones_like(t)),
                create_graph=False,
            )
        return dict(diagonal=state["tokens"], consistency=(value + tangent).detach())


class AffineMapSampler:
    def __init__(self, steps=4):
        if steps < 1 or int(steps) != steps:
            raise ValueError("steps must be a positive integer")
        self.steps = int(steps)

    def steps_for_nfe(self, nfe):
        return nfe

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        clean = algo.view.representation.encode(prompt_tokens)
        time = torch.ones(len(clean), device=clean.device)
        state = algo.view.process.prior_state(clean, algo.coefficients(time))
        times = torch.linspace(1, 0, self.steps + 1, device=clean.device)
        for i in range(self.steps):
            state = torch.where(prompt_mask[..., None], clean, state)
            source = (
                time * times[i]
                if not algo.endpoint or i == 0
                else torch.zeros_like(time)
            )
            destination = (
                time * times[i + 1] if not algo.endpoint else torch.zeros_like(time)
            )
            probabilities = algo(state, source, destination).exp()
            state = (
                algo.view.representation.posterior_mean(probabilities)
                if algo.endpoint
                else algo.transport(state, probabilities, source, destination)
            )
        tokens = torch.where(prompt_mask, prompt_tokens, probabilities.argmax(-1))
        return tokens, dict(nfe=self.steps, nfe_per_sample=[self.steps] * len(tokens))
