"""Continuous sampling lifecycle, independent of the integration method."""

import math
from contextlib import nullcontext
import torch

from core.parameterizations.posterior import PosteriorParameterization
from core.modifiers.self_conditioning.session import (
    PredictionSession,
    prediction_session,
)
from .loop import FixedGridLoop, LinearTimeGrid
from .decoding import PosteriorDecoding, RepresentationDecoding


class PosteriorMeanVelocity:
    """Obtain a clean-state mean, then let the process convert it to velocity."""

    def __call__(self, algo, state, time):
        path = algo.schedule(time)
        mean = algo.posterior_mean(state, time, path)
        return algo.process.reverse_vector_field(
            state, mean, path.alpha, path.sigma, path.dalpha_dt, path.dsigma_dt
        )


class ContinuousSampler:
    supports_prediction_feedback = False

    def __init__(self, epsilon=1e-5, start_time=0.999, decoding="posterior"):
        if not (
            math.isfinite(epsilon)
            and math.isfinite(start_time)
            and 0 < epsilon < start_time < 1
        ):
            raise ValueError("require finite 0 < epsilon < start_time < 1")
        self.epsilon, self.start_time = epsilon, start_time
        self.velocity = PosteriorMeanVelocity()
        decoders = {
            "posterior": PosteriorDecoding,
            "representation": RepresentationDecoding,
        }
        if decoding not in decoders:
            raise ValueError("decoding must be posterior or representation")
        self.decoder = decoders[decoding]()

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if not isinstance(getattr(algo, "predictor", None), PosteriorParameterization):
            raise TypeError("continuous sampling requires PosteriorParameterization")
        algo = prediction_session(algo)
        if (
            isinstance(algo, PredictionSession)
            and not self.supports_prediction_feedback
        ):
            raise ValueError(
                "self-conditioning is not yet integrated with adaptive ODE sampling; use Euler or Heun"
            )
        initial_time = torch.full(
            (len(prompt_tokens),), self.start_time, device=prompt_tokens.device
        )
        clean = algo.representation.encode(prompt_tokens)
        fixed = prompt_mask.bool()[..., None]

        def constrain(state):
            return torch.where(fixed, clean, state)

        state = constrain(algo.process.prior(clean, algo.schedule(initial_time).sigma))
        nfe = 0

        def field(time, state):
            nonlocal nfe
            if hasattr(self, "max_nfe") and nfe >= self.max_nfe - 1:
                raise RuntimeError(
                    f"continuous sampler exceeded max_nfe={self.max_nfe} "
                    "(including final decoding)"
                )
            nfe += 1
            value = self.velocity(algo, constrain(state), time.expand(len(state)))
            if value.shape != state.shape or not torch.isfinite(value).all():
                raise ValueError("velocity must have the state shape and finite values")
            return torch.where(fixed, torch.zeros_like(value), value)

        if isinstance(algo, PredictionSession):
            field.step_context = algo.step
        state = constrain(self.integrate(field, state))
        if not torch.isfinite(state).all():
            raise ValueError("continuous integration produced a nonfinite state")
        time = initial_time.new_full(initial_time.shape, self.epsilon)
        tokens = self.decoder(algo, state, time)
        nfe += 1
        tokens = torch.where(prompt_mask.bool(), prompt_tokens, tokens)
        return tokens, {"nfe": nfe, "nfe_per_sample": [nfe] * len(tokens)}

    def integrate(self, field, state):
        raise NotImplementedError


class FixedStepSampler(ContinuousSampler):
    supports_prediction_feedback = True

    def __init__(self, steps=180, update_rule=None, time_grid=None, **kwargs):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        super().__init__(**kwargs)
        self.steps = steps
        self.update_rule = update_rule
        self.time_grid = time_grid or LinearTimeGrid(
            self.start_time, self.epsilon, steps
        )
        self.loop = FixedGridLoop()

    def integrate(self, field, state):
        times = self.time_grid(state.device)
        if not (
            torch.isclose(times[0], times.new_tensor(self.start_time))
            and torch.isclose(times[-1], times.new_tensor(self.epsilon))
        ):
            raise ValueError(
                "integration grid must preserve start_time and epsilon endpoints"
            )
        return self.loop(
            state,
            times,
            lambda value, time, next_time, index: self.step(
                field, value, time, next_time
            ),
            step_context=getattr(field, "step_context", nullcontext),
        )

    def step(self, field, state, time, next_time):
        if self.update_rule is None:
            raise NotImplementedError("provide an update_rule or override step")
        return self.update_rule(field, state, time, next_time)
