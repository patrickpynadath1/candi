"""Uniform diffusion with normalized plug-in estimates of clean token weights.

Runs on the view path: one uniform view on a ModularDiT adopted from the
configured DiT (see core/models/legacy.py:adopt). The former standalone class
is kept in core/tests/reference_algorithms/uniform_diffusion.py, and
core/tests/test_view_equivalence_uniform.py proves both behave the same.
"""

import torch

from core.algorithms.adaptive import validate_adaptive_feedback
from core.models.legacy import adopt
from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.objectives.cross_entropy import CrossEntropy
from core.objectives.uniform import UniformDiffusionObjective
from core.objectives.weighting import ObjectiveContext
from core.parameterizations.plug_in import PlugInParameterization
from core.processes.uniform import UniformProcess
from core.representations.noisy_inputs import TokenRepresentation
from core.samplers.hybrid_updates import DiscretePosteriorUpdate


class UniformDiffusion(OneViewAlgorithm, HybridDiffusion):
    prediction_pathway = "plugin"
    prediction_kind = "plug_in"
    # The token view's input map; Duo swaps in one that also embeds soft states.
    representation_type = TokenRepresentation

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
        time_conditioning=True,
    ):
        prediction = prediction if prediction is not None else PredictionCoordinator()
        validate_adaptive_feedback(schedule, predictor)
        if getattr(schedule, "output_kind", None) != "keep_rate":
            raise TypeError("discrete diffusion requires a keep-rate noise schedule")
        if not model.adaLN:
            raise ValueError(
                "this discrete method requires a time-conditioned model (adaLN=true)"
            )
        prediction.validate(model, predictor, training_pathway="training_plugin")
        if not isinstance(predictor, PlugInParameterization):
            raise TypeError("UniformDiffusion requires PlugInParameterization")
        if not isinstance(process, UniformProcess) or not isinstance(
            objective, UniformDiffusionObjective
        ):
            raise TypeError("UniformDiffusion requires a uniform process and objective")
        if (
            process.vocab_size != model.output_layer.linear.out_features
            or objective.vocab_size != process.vocab_size
        ):
            raise ValueError(
                "uniform model, process and objective vocabulary sizes must agree"
            )
        table = model.vocab_embed.embedding
        with torch.random.fork_rng(devices=[]):  # The table is adopted below.
            representation = self.representation_type(*table.shape)
        view = DiffusionView(process, schedule, representation, DiscretePosteriorUpdate())
        network = adopt(
            model,
            views=[view],
            # -log(keep), keep clamped at the dtype's smallest normal, or zero.
            time_coordinate="negative_log_keep_exact" if time_conditioning else "zero",
            fp32_clean_embedding=True,
        )
        super().__init__(
            network, predictor, time_sampler, objective, length=length, prediction=prediction
        )
        self.time_conditioning = time_conditioning

    def _objective(self, objective, prediction, tokens, time, points, states):
        keep, derivative = points[0].coefficients
        if isinstance(objective, CrossEntropy):
            # Per-call CE (block-size mixtures) sees the keep-rate path.
            return objective(
                prediction,
                tokens,
                context=ObjectiveContext(
                    time=time, alpha=keep, sigma=-keep.clamp_min(1e-30).log()
                ),
            )
        return objective(prediction, states[0], tokens, keep[:, None], derivative[:, None])
