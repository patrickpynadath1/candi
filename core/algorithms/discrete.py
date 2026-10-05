"""Shared token-diffusion training lifecycle, without sampling mathematics."""

from .adaptive import validate_adaptive_feedback, observe_training_loss

import torch
from torch import nn


def reduce_token_loss(per_token, loss_mask):
    weights = loss_mask.to(per_token)
    count = weights.sum()
    if count <= 0:
        raise ValueError("loss mask must include at least one target")
    total = (per_token * weights).sum()
    return {
        "loss": total / count,
        "loss_sum": total.detach(),
        "num_tokens": count.detach(),
    }


class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        prediction=None,
    ):
        super().__init__()
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.length = length
        self.prediction = prediction
        if (
            prediction is None
            and getattr(model, "conditioning_adapter", None) is not None
        ):
            raise ValueError(
                "self-conditioning prediction is not yet integrated for this algorithm"
            )
        validate_adaptive_feedback(schedule, predictor)
        if getattr(schedule, "output_kind", None) != "keep_rate":
            raise TypeError("discrete diffusion requires a keep-rate noise schedule")
        if not model.adaLN:
            raise ValueError(
                "this discrete method requires a time-conditioned model (adaLN=true)"
            )

    def model_logits(self, state, keep):
        return self.model(state, -keep.log())

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        **time_context,
    ):
        if corruption_mask is None:
            corruption_mask = loss_mask
        times = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        keep, derivative = self.schedule(times)
        state = self.process.corrupt(tokens, keep, corruption_mask)
        prediction = self.training_prediction(state, keep)
        per_token = self.objective(
            prediction, state, tokens, keep[:, None], derivative[:, None]
        )
        observe_training_loss(
            self.schedule,
            times,
            prediction,
            tokens,
            per_token,
            loss_mask,
            step=step,
            training=training,
        )
        return reduce_token_loss(per_token, loss_mask)

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)
