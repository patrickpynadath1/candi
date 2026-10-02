"""Shifted-token training and cached model-to-predictor communication."""

import torch
from torch import nn
from .discrete import reduce_token_loss
from core.models.causal import CausalDiT
from core.parameterizations.posterior import PosteriorParameterization


class Autoregressive(nn.Module):
    def __init__(self, model, predictor, objective, length=180):
        super().__init__()
        if not isinstance(model, CausalDiT) or not isinstance(
            predictor, PosteriorParameterization
        ):
            raise TypeError(
                "autoregressive requires a causal model and token posterior"
            )
        self.model, self.predictor, self.objective = model, predictor, objective
        self.length = length

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
        if tokens.shape[1] < 2:
            raise ValueError("autoregressive training requires at least two tokens")
        prediction = self.predictor.training_posterior(self.model(tokens[:, :-1]), None)
        return reduce_token_loss(
            self.objective(prediction, tokens[:, 1:]), loss_mask[:, 1:]
        )

    def sampling_prediction(self, tokens, cache=None):
        return self.predictor.sampling_posterior(self.model(tokens, cache=cache), None)

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)
