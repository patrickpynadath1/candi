"""Shared posterior prediction endpoints, separate from training objectives."""

import torch
from torch import nn
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.modifiers.self_conditioning.sources import posterior_mean


class PosteriorDiffusion(nn.Module):
    # Parameterization endpoints are named "<use>_<pathway>", e.g.
    # training_posterior or sampling_plugin.
    prediction_pathway = "posterior"

    def __init__(self, prediction=None):
        super().__init__()
        self.prediction = (
            prediction if prediction is not None else PredictionCoordinator()
        )

    def validate_prediction(self):
        self.prediction.validate(
            self.model, self.predictor, training_pathway=self._pathway("training")
        )

    def _pathway(self, use):
        return f"{use}_{self.prediction_pathway}"

    def model_logits(self, *state):
        return self.model(*self.model_inputs(*state))

    def prediction_embedding_table(self):
        return self.model.token_table

    def prediction_mean(self, probabilities):
        return posterior_mean(probabilities, self.prediction_embedding_table())

    def prediction_logits(self, logits):
        return logits

    def _prediction(self, pathway, *state, trajectory=None):
        return self.prediction.predict(
            self.model,
            self.predictor,
            self.model_inputs(*state),
            state[0],
            pathway,
            mean=self.prediction_mean,
            logits_transform=self.prediction_logits,
            trajectory=trajectory,
            training_pathway=self._pathway("training"),
        )

    def training_prediction(self, *state):
        return self._prediction(self._pathway("training"), *state)

    def sampling_prediction(self, *state, trajectory=None):
        return self._prediction(self._pathway("sampling"), *state, trajectory=trajectory)

    def greedy_prediction(self, *state, trajectory=None):
        return self._prediction(self._pathway("greedy"), *state, trajectory=trajectory)

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)
