"""MDLM training orchestration and model-to-posterior prediction."""

from .adaptive import validate_adaptive_feedback, observe_training_loss

import torch
from .posterior import PosteriorDiffusion


class MDLM(PosteriorDiffusion):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        time_conditioning=False,
        prediction=None,
    ):
        super().__init__(prediction)
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.length, self.time_conditioning = length, time_conditioning
        validate_adaptive_feedback(schedule, predictor)
        self.validate_prediction()

    def model_inputs(self, state, keep):
        condition = -keep.log() if self.time_conditioning else torch.zeros_like(keep)
        return state, condition

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
        noisy = self.process.corrupt(tokens, keep, corruption_mask)
        prediction = self.training_prediction(
            noisy, keep
        )  # Shaping cannot enter this path.
        per_token = self.objective(prediction, tokens, keep, derivative)
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
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
        total = (per_token * weights).sum()
        return {
            "loss": total / count,
            "loss_sum": total.detach(),
            "num_tokens": count.detach(),
        }

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)
