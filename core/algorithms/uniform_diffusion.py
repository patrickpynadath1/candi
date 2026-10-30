"""Uniform diffusion with normalized plug-in estimates of clean token weights."""

from .discrete import DiscreteDiffusion
import torch
from core.parameterizations.plug_in import PlugInParameterization
from core.processes.uniform import UniformProcess
from core.objectives.uniform import UniformDiffusionObjective
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.modifiers.self_conditioning.sources import posterior_mean


class UniformDiffusion(DiscreteDiffusion):
    def __init__(self, *args, prediction=None, time_conditioning=True, **kwargs):
        super().__init__(
            *args,
            prediction=prediction
            if prediction is not None
            else PredictionCoordinator(),
            **kwargs,
        )
        self.time_conditioning = time_conditioning
        self.prediction.validate(
            self.model, self.predictor, training_pathway="training_plugin"
        )
        if not isinstance(self.predictor, PlugInParameterization):
            raise TypeError("UniformDiffusion requires PlugInParameterization")
        if not isinstance(self.process, UniformProcess) or not isinstance(
            self.objective, UniformDiffusionObjective
        ):
            raise TypeError("UniformDiffusion requires a uniform process and objective")
        if (
            self.process.vocab_size != self.model.output_layer.linear.out_features
            or self.objective.vocab_size != self.process.vocab_size
        ):
            raise ValueError(
                "uniform model, process and objective vocabulary sizes must agree"
            )

    def prediction_mean(self, probabilities):
        return posterior_mean(probabilities, self.model.vocab_embed.embedding)

    def _prediction(self, pathway, state, keep, trajectory=None):
        return self.prediction.predict(
            self.model,
            self.predictor,
            (state, -keep.clamp_min(torch.finfo(keep.dtype).tiny).log() if self.time_conditioning else torch.zeros_like(keep)),
            state,
            pathway,
            mean=self.prediction_mean,
            trajectory=trajectory,
            training_pathway="training_plugin",
        )

    def training_prediction(self, state, keep):
        return self._prediction("training_plugin", state, keep)

    def sampling_prediction(self, state, keep, trajectory=None):
        return self._prediction("sampling_plugin", state, keep, trajectory)

    def greedy_prediction(self, state, keep, trajectory=None):
        return self._prediction("greedy_plugin", state, keep, trajectory)
