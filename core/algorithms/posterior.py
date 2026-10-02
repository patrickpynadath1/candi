"""Shared posterior prediction endpoints, separate from training objectives."""

from torch import nn
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.modifiers.self_conditioning.sources import posterior_mean


class PosteriorDiffusion(nn.Module):
    def __init__(self, prediction=None):
        super().__init__()
        self.prediction = (
            prediction if prediction is not None else PredictionCoordinator()
        )

    def validate_prediction(self):
        self.prediction.validate(self.model, self.predictor)

    def model_logits(self, *state):
        return self.model(*self.model_inputs(*state))

    def prediction_embedding_table(self):
        return self.model.vocab_embed.embedding

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
        )

    def training_prediction(self, *state):
        return self._prediction("training_posterior", *state)

    def sampling_prediction(self, *state, trajectory=None):
        return self._prediction("sampling_posterior", *state, trajectory=trajectory)

    def greedy_prediction(self, *state, trajectory=None):
        return self._prediction("greedy_posterior", *state, trajectory=trajectory)
