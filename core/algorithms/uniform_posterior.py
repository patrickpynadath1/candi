"""Uniform corruption trained as clean-token posterior prediction with CE.

Runs on the view path like MDLM: one uniform-posterior view on a ModularDiT
adopted from the configured DiT (core/models/legacy.py:adopt). Training and
sampling match the former standalone implementation, which
core/tests/reference_algorithms keeps for comparison; its checkpoints load
through core/models/legacy_checkpoints.py.
"""

import torch

from core.models.legacy import adopt
from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.objectives.cross_entropy import CrossEntropy
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.constraints import UnconstrainedPosterior
from core.processes.uniform_posterior import PosteriorUniformProcess
from core.modifiers.hybrid.views import DiffusionView
from core.representations.noisy_inputs import TokenRepresentation
from core.samplers.hybrid_updates import DiscretePosteriorUpdate


class PosteriorUniformDiffusion(OneViewAlgorithm, HybridDiffusion):
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
        table = model.vocab_embed.embedding
        with torch.random.fork_rng(devices=[]):  # The table is adopted below.
            representation = TokenRepresentation(*table.shape)
        view = DiffusionView(process, schedule, representation, DiscretePosteriorUpdate())
        # -log(keep) exactly as the standalone class computed it.
        network = adopt(model, views=[view], time_coordinate="negative_log_keep_exact")
        super().__init__(
            network,
            predictor,
            time_sampler,
            objective,
            length=length,
            prediction=prediction,
        )
