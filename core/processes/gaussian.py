"""Affine Gaussian corruption of continuous representations."""

import torch


class GaussianProcess:
    def forward_distribution(self, clean, coefficients):
        if torch.any(coefficients.sigma <= 0):
            raise ValueError("Gaussian density requires positive sigma")
        mean = coefficients.alpha[:, None, None] * clean
        scale = coefficients.sigma[:, None, None].expand_as(mean)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, scale), 1
        )

    def corrupt_state(self, clean, coefficients, corruption_mask):
        return self.corrupt(
            clean, coefficients.alpha, coefficients.sigma, corruption_mask
        )

    def prior_state(self, clean, coefficients):
        return self.prior(clean, coefficients.sigma)

    def corrupt(self, clean, alpha, sigma, corruption_mask):
        noisy = (
            alpha.to(clean)[:, None, None] * clean
            + torch.randn_like(clean) * sigma.to(clean)[:, None, None]
        )
        return torch.where(corruption_mask.bool()[..., None], noisy, clean)

    def prior(self, clean, sigma):
        return torch.randn_like(clean) * sigma.to(clean)[:, None, None]

    def reverse_vector_field(
        self, state, posterior_mean, alpha, sigma, dalpha_dt, dsigma_dt
    ):
        """Time velocity for an affine Gaussian path, given its clean-state mean."""
        if torch.any(sigma <= 0):
            raise ValueError(
                "reverse field requires positive sigma; stop before the clean endpoint"
            )
        residual = state - alpha[:, None, None] * posterior_mean
        return (
            dalpha_dt[:, None, None] * posterior_mean
            + (dsigma_dt / sigma)[:, None, None] * residual
        )

    def reverse_posterior(self, state, clean, path_t, path_s, noise):
        """Sample the affine-Gaussian reverse bridge using a clean-state estimate.

        Works for VE, VP and interpolating paths with nondecreasing reverse
        SNR. The posterior-mean estimate is supplied by the representation;
        this process does not depend on tokens or embedding dimensionality.
        Zero-alpha (pure noise) and zero-sigma (clean) endpoints are supported.
        """
        dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
        alpha_t, sigma_t, alpha_s, sigma_s = (
            value.to(device=state.device, dtype=dtype)
            for value in (path_t.alpha, path_t.sigma, path_s.alpha, path_s.sigma)
        )
        for alpha, sigma in ((alpha_t, sigma_t), (alpha_s, sigma_s)):
            if torch.any(~torch.isfinite(alpha) | ~torch.isfinite(sigma)):
                raise ValueError("Gaussian reverse coefficients must be finite")
            if torch.any((alpha < 0) | (sigma < 0) | ((alpha == 0) & (sigma == 0))):
                raise ValueError(
                    "Gaussian reverse coefficients must be nonnegative and nondegenerate"
                )
        # Forward conditional gain alpha_t / alpha_s. A segment with both
        # alphas zero has constant zero SNR; choose deterministic noise scaling.
        pure_noise = alpha_s == 0
        if torch.any(pure_noise & (alpha_t > 0)):
            raise ValueError(
                "reverse posterior requires nondecreasing reverse-time SNR"
            )
        clean_source = sigma_t == 0
        if torch.any(clean_source & (sigma_s > 0)):
            raise ValueError(
                "reverse posterior requires nondecreasing reverse-time SNR"
            )
        safe_alpha_s = torch.where(pure_noise, torch.ones_like(alpha_s), alpha_s)
        safe_sigma_t = torch.where(clean_source, torch.ones_like(sigma_t), sigma_t)
        forward_gain = alpha_t / safe_alpha_s
        sigma_ratio = sigma_s / safe_sigma_t
        correlation = forward_gain * sigma_ratio
        # expm1 avoids cancellation for tiny steps. Unlike alpha_s/alpha_t,
        # this expression also remains defined at alpha_t=0.
        c = -torch.expm1(2 * correlation.log())
        c = torch.where(pure_noise | clean_source, torch.zeros_like(c), c)
        if torch.any(c < -1e-6):
            raise ValueError(
                "reverse posterior requires nondecreasing reverse-time SNR"
            )
        c = c.clamp(0, 1)
        state_gain = correlation * sigma_ratio
        state_gain = torch.where(pure_noise, sigma_ratio, state_gain)
        state_gain = torch.where(
            clean_source,
            alpha_s / torch.where(alpha_t == 0, torch.ones_like(alpha_t), alpha_t),
            state_gain,
        )
        shape = (-1,) + (1,) * (state.ndim - 1)
        result = (
            state_gain.reshape(shape) * state.to(dtype)
            + (c * alpha_s).reshape(shape) * clean.to(dtype)
            + (c.sqrt() * sigma_s).reshape(shape) * noise.to(dtype)
        )
        return result.to(state.dtype)
