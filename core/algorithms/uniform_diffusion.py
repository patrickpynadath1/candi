"""Uniform diffusion with normalized plug-in estimates of clean token weights."""

from .discrete import DiscreteDiffusion
from core.parameterizations.plug_in import PlugInParameterization
from core.processes.uniform import UniformProcess
from core.objectives.uniform import UniformDiffusionObjective


class UniformDiffusion(DiscreteDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def training_prediction(self, state, keep):
        return self.predictor.training_plugin(self.model_logits(state, keep))

    def sampling_prediction(self, state, keep):
        return self.predictor.sampling_plugin(self.model_logits(state, keep))

    def greedy_prediction(self, state, keep):
        return self.predictor.greedy_plugin(self.model_logits(state, keep))
