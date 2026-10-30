"""Composable posterior-driven updates for individual noisy views.

Each update consumes its own process/state and the one shared token posterior.
"""

import torch


class DiscretePosteriorUpdate:
    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        return process.reverse(
            state, probabilities, source.coefficients[0], destination.coefficients[0]
        )


class GaussianPosteriorUpdate:
    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        clean = representation.posterior_mean(probabilities)
        noise = torch.randn_like(state) if noise is None else noise
        return process.reverse_posterior(
            state, clean, source.coefficients, destination.coefficients, noise
        )


class SimplexPosteriorUpdate:
    def __init__(self, churn=1.0):
        self.churn = float(churn)

    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        clean = torch.distributions.Categorical(probabilities).sample()
        return process.reverse(
            state, clean, destination.coefficients, source.coefficients, self.churn
        )


class SphericalPosteriorUpdate:
    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        from .spherical import posterior_velocity

        dtype = torch.float64 if process.precision == "float64" else torch.float32
        velocity = posterior_velocity(
            state.to(dtype),
            representation.codebook().to(dtype),
            probabilities.to(dtype),
            eps=process.eps,
        )
        keep, next_keep = source.coefficients[0], destination.coefficients[0]
        delta = ((next_keep - keep) / (1 - keep).clamp_min(process.eps))[:, None, None]
        return process.advance(state.to(dtype), delta * velocity).to(state.dtype)


class GaussianVelocityEulerUpdate:
    """Probability-flow Euler step from the shared token posterior."""

    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        path = source.coefficients
        velocity = process.reverse_vector_field(
            state,
            representation.posterior_mean(probabilities),
            path.alpha,
            path.sigma,
            path.dalpha_dt,
            path.dsigma_dt,
        )
        return state + (destination.time - source.time)[:, None, None] * velocity


class BlurringPosteriorUpdate:
    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        return process.reverse(
            state,
            representation.posterior_mean(probabilities),
            source.coefficients,
            destination.coefficients,
            noise=noise,
        )


class GIDDUpdate:
    """GIDD marginal reverse update, including revisable non-mask tokens."""

    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        return process.reverse(
            state, probabilities, source.coefficients, destination.coefficients
        )


class VMFProbabilityFlowUpdate:
    """Paper Algorithm 2 predictor: tangent Euler step and normalization."""

    def __call__(
        self,
        process,
        representation,
        state,
        probabilities,
        source,
        destination,
        *,
        noise=None,
    ):
        from core.processes.spherical import sphere_normalize

        velocity = process.velocity(
            state, representation.codebook(), probabilities, source.coefficients[0]
        )
        # Euler in local time: the derivative belongs to the source schedule.
        delta = (destination.time - source.time) * source.coefficients[1]
        while delta.ndim < state.ndim:
            delta = delta.unsqueeze(-1)
        return sphere_normalize(state.float() + delta * velocity).to(state.dtype)
