"""SEDD absorbing training and model-to-score orchestration."""

from .discrete import DiscreteDiffusion
from core.parameterizations.score import AbsorbingScoreParameterization
from core.processes.absorbing import AbsorbingProcess
from core.objectives.score_entropy import AbsorbingScoreEntropy


class SeddAbsorb(DiscreteDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.predictor, AbsorbingScoreParameterization):
            raise TypeError("SeddAbsorb requires AbsorbingScoreParameterization")
        if not isinstance(self.process, AbsorbingProcess) or not isinstance(
            self.objective, AbsorbingScoreEntropy
        ):
            raise TypeError(
                "SeddAbsorb requires an absorbing process and score-entropy objective"
            )
        if (
            self.objective.mask_index != self.process.mask_index
            or self.objective.vocab_size != self.model.output_layer.linear.out_features
        ):
            raise ValueError(
                "SEDD model, process and objective token spaces must agree"
            )

    def training_prediction(self, state, keep):
        return self.predictor(self.model_logits(state, keep), state, -keep.log())

    def sampling_prediction(self, state, keep):
        return self.training_prediction(state, keep)
