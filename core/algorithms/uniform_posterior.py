"""Uniform corruption trained as clean-token posterior prediction with CE.

Runs on the hybrid path as one view (see core/modifiers/hybrid/one_view.py);
training, sampling and checkpoints are bit-identical to the former standalone
implementation, which core/tests/reference_algorithms keeps for comparison.
"""

from core.modifiers.hybrid.one_view import one_view_model
from core.modifiers.hybrid.training import HybridDiffusion
from core.objectives.cross_entropy import CrossEntropy
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.constraints import UnconstrainedPosterior
from core.processes.uniform_posterior import PosteriorUniformProcess


class PosteriorUniformDiffusion(HybridDiffusion):
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
        if (
            not isinstance(predictor, PosteriorParameterization)
            or not isinstance(predictor.constraints, UnconstrainedPosterior)
            or predictor.mask_index is not None
        ):
            raise TypeError(
                "posterior uniform diffusion requires an unconstrained posterior; noisy tokens are not revealed clean tokens"
            )
        if not isinstance(process, PosteriorUniformProcess):
            raise TypeError(
                "posterior uniform diffusion requires PosteriorUniformProcess"
            )
        if not isinstance(objective, CrossEntropy) or objective.requires_context:
            raise TypeError(
                "posterior uniform diffusion requires cross entropy without Gaussian path weighting"
            )
        if getattr(schedule, "output_kind", None) != "keep_rate":
            raise TypeError("posterior uniform diffusion requires a keep-rate schedule")
        if not model.adaLN:
            raise ValueError(
                "posterior uniform diffusion requires time conditioning (adaLN=true)"
            )
        if process.vocab_size != model.output_layer.linear.out_features:
            raise ValueError("model and process vocabulary sizes must agree")
        super().__init__(
            one_view_model(model, process, schedule),
            predictor,
            time_sampler,
            objective,
            length=length,
            prediction=prediction,
        )

    @property
    def process(self):
        return self.views[0].process

    @property
    def schedule(self):
        return self.views[0].schedule
