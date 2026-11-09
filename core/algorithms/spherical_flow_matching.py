"""Spherical flow matching: unit-sphere embeddings, slerp corruption, CE posterior.

One spherical view held by the algorithm. The network reads -log(keep) with keep
clamped at 1e-12 from the view's point.
"""

from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView
from core.representations.embedding import EmbeddingRepresentation
from core.samplers.hybrid_updates import SphericalPosteriorUpdate


class SphericalFlowMatching(OneViewAlgorithm, HybridDiffusion):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        representation,
        length=180,
        prediction=None,
    ):
        if (
            not isinstance(representation, EmbeddingRepresentation)
            or not representation.spec.unit_normalized
        ):
            raise ValueError(
                "spherical flow matching requires unit-normalized embeddings"
            )
        if (
            model.state_dim != representation.spec.dimension
            or model.vocab_size != representation.spec.vocab_size
        ):
            raise ValueError("model and representation dimensions must match")
        if getattr(schedule, "output_kind", None) != "keep_rate":
            raise ValueError("spherical schedule must provide keep coordinates")
        view = DiffusionView(process, schedule, representation, SphericalPosteriorUpdate())
        model.time.coordinate = "negative_log_keep_floor"
        super().__init__(
            model,
            predictor,
            time_sampler,
            objective,
            length=length,
            prediction=prediction,
            views=[view],
        )

    def prediction_logits(self, logits):
        return logits.float()

    def prediction_mean(self, probabilities):
        # Expected embedding in state space, deliberately not sphere-normalized.
        return self.representation.posterior_mean(probabilities)

    def _objective(self, objective, prediction, tokens, time, points, states, **entries):
        # Plain per-token CE: the spherical path has no Gaussian alpha/sigma.
        return objective(prediction, tokens)
