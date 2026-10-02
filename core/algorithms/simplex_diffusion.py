"""Simplex training orchestration; process and sampler own transition mathematics."""

import torch
from .posterior import PosteriorDiffusion


class SimplexDiffusion(PosteriorDiffusion):
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
        condition_dropout=0.0,
    ):
        super().__init__(prediction)
        if not 0 <= condition_dropout <= 1:
            raise ValueError("condition_dropout must be in [0,1]")
        if model.vocab_size != process.vocab_size:
            raise ValueError("simplex process and model vocabulary must match")
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.length, self.condition_dropout = length, condition_dropout
        self.validate_prediction()

    def model_inputs(self, state, time):
        return state, time

    def prediction_logits(self, logits):
        return logits.float()

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
        time = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        state = self.process.corrupt(tokens, self.schedule(time), corruption_mask)
        if training:
            state = self.process.drop_condition(
                state, corruption_mask, self.condition_dropout
            )
        prediction = self.training_prediction(state, time)
        per_token = self.objective(prediction, tokens)
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        observe = getattr(self.time_sampler, "observe", None)
        if training and observe is not None:
            observe(time, per_token, loss_mask, step)
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
