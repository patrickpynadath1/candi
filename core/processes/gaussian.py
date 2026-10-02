"""Affine Gaussian corruption of continuous representations."""

import torch


class GaussianProcess:
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
        """Gaussian q(z_s | z_t, x0), for any positive monotone-SNR path."""
        alpha_t = path_t.alpha.to(state)
        alpha_s = path_s.alpha.to(state)
        sigma_s = path_s.sigma.to(state)
        log_snr_t = 2 * (path_t.alpha.log() - path_t.sigma.log())
        log_snr_s = 2 * (path_s.alpha.log() - path_s.sigma.log())
        c = -torch.expm1(log_snr_t - log_snr_s).to(state)
        if torch.any(c < -1e-6):
            raise ValueError(
                "reverse posterior requires nondecreasing reverse-time SNR"
            )
        c = c.clamp_min(0)[:, None, None]
        ratio = (alpha_s / alpha_t)[:, None, None]
        return (
            ratio * (1 - c) * state
            + c * alpha_s[:, None, None] * clean
            + c.sqrt() * sigma_s[:, None, None] * noise
        )
