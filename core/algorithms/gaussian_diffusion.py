"""Gaussian diffusion over discrete representations, as a one-view recipe.

One Gaussian view held by the algorithm. The schedule's training-time coordinate
(``schedule.training_time``) feeds the network, the objective context and the time
sampler's feedback.
"""

from dataclasses import replace

from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView, ViewPoint
from core.representations.one_hot import OneHotRepresentation
from core.samplers.hybrid_updates import GaussianVelocityEulerUpdate


class GaussianDiffusion(OneViewAlgorithm, HybridDiffusion):
    # Continuous states carry no observed tokens to clamp.
    clamp_observed = False

    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        representation=None,
        prediction=None,
    ):
        # The default also reconstructs saved pre-representation one-hot configs.
        if representation is None:
            representation = OneHotRepresentation(model.vocab_size)
        spec = representation.spec
        if model.state_dim != spec.dimension:
            raise ValueError("model state_dim must match representation dimension")
        if model.vocab_size != spec.vocab_size:
            raise ValueError("model vocabulary must match representation vocabulary")
        # The view validates the schedule's representation requirements.
        view = DiffusionView(process, schedule, representation, GaussianVelocityEulerUpdate())
        network = model
        # The network reads the view's local time: in training, the schedule's
        # training-time coordinate, as the standalone class fed it.
        network.time.time_source = "point"
        super().__init__(
            network,
            predictor,
            time_sampler,
            objective,
            length=length,
            prediction=prediction,
            views=[view],
        )

    def prediction_mean(self, probabilities):
        # Feedback lives in state space; the network applies its scale.
        return self.representation.posterior_mean(probabilities)

    def posterior_mean(self, state, time, path):
        """Clean-state mean of the sampling posterior at ``(time, path)``."""
        probabilities = (
            self.sampling_prediction((state,), (ViewPoint(time, path),), time).float().exp()
        )
        return self.representation.posterior_mean(probabilities)

    def _objective_context(self, time, points, states):
        # Objectives see the training-time coordinate the network saw.
        return replace(super()._objective_context(time, points, states), time=points[0].time)

    def _observe(self, sampler, feedback_time, time, points, *rest):
        # The time sampler observes the schedule's training-time coordinate,
        # as in existing adaptive Gaussian runs (a <=training_epsilon rescaling).
        super()._observe(sampler, points[0].time, time, points, *rest)
