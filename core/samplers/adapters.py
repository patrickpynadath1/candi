"""Sampling-only views over existing algorithms; no new model parameters."""

from types import SimpleNamespace

from core.modifiers.hybrid.views import ViewPoint
from .hybrid_updates import DiscretePosteriorUpdate


class TokenState:
    """Identity clean encoding for a process whose states are token IDs."""

    @staticmethod
    def encode(tokens):
        return tokens

    clean_state = encode


class KeepRateSamplerAdapter:
    """Translate the shared view API into (token state, keep rate) predictions.

    Preserve the original process, schedule and prediction coordinator. This
    wrapper is ephemeral: it neither registers parameters nor changes checkpoints.
    """

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.views = (
            SimpleNamespace(
                process=algorithm.process,
                representation=TokenState(),
                update=DiscretePosteriorUpdate(),
            ),
        )

    def __getattr__(self, name):
        return getattr(self.algorithm, name)

    def points(self, time, *, corruption_mask=None):
        return (ViewPoint(time, self.algorithm.schedule(time)),)

    def sampling_prediction(self, states, points, time, *, trajectory=None):
        return self.algorithm.sampling_prediction(
            states[0], points[0].coefficients[0], trajectory=trajectory
        )

    def greedy_prediction(self, states, points, time, *, trajectory=None):
        return self.algorithm.greedy_prediction(
            states[0], points[0].coefficients[0], trajectory=trajectory
        )
