"""Sampler-owned, local prediction state; no mutation of the algorithm or model."""

from contextlib import contextmanager

from .prediction import TrajectoryState


class PredictionSession:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.state = TrajectoryState()

    def __getattr__(self, name):
        return getattr(self.algorithm, name)

    def sampling_prediction(self, *args):
        return self.algorithm.sampling_prediction(*args, trajectory=self.state)

    def greedy_prediction(self, *args):
        return self.algorithm.greedy_prediction(*args, trajectory=self.state)

    def posterior_mean(self, *args):
        probabilities = self.sampling_prediction(*args).float().exp()
        return self.algorithm.representation.posterior_mean(probabilities)

    @contextmanager
    def step(self):
        """Freeze feedback across solver stages, then commit after a successful step."""
        if self.state.staged:
            raise RuntimeError("prediction stages cannot be nested")
        self.state.staged = True
        self.state.pending = None
        try:
            yield
        except BaseException:
            raise
        else:
            if self.state.pending is not None:
                self.state.conditioning = self.state.pending
        finally:
            self.state.staged = False
            self.state.pending = None


def prediction_session(algorithm):
    while isinstance(algorithm, PredictionSession):
        algorithm = algorithm.algorithm
    policy = getattr(algorithm, "prediction", None)
    if policy is None or not policy.enabled:
        return algorithm
    return PredictionSession(algorithm)
