"""Sequence-adaptive feature blur; no mixing along the position axis."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PCABasis:
    vectors: torch.Tensor
    eigenvalues: torch.Tensor
    rates: torch.Tensor


class PCABlurringProcess:
    """Detached FP32 covariance PCA on active positions, with isotropic noise.

    The scalar schedule supplies alpha in [0, 1] and nonnegative sigma.
    Direction i retains alpha ** rate_i. All rates are >= 1, including null
    directions. Centering is used ONLY for covariance, never for corruption.
    """

    def __init__(self, strength=3.0, ridge=1e-6):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("PCA strength must be finite and nonnegative")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("PCA ridge must be finite and positive")
        self.strength, self.ridge = float(strength), float(ridge)

    @torch.no_grad()
    def basis(self, clean, corruption_mask):
        if clean.ndim != 3 or corruption_mask.shape != clean.shape[:2]:
            raise ValueError(
                "expected clean [batch, length, features] and mask [batch, length]"
            )
        with torch.autocast(clean.device.type, enabled=False):
            x = clean.detach().float()
            mask = corruption_mask.bool()[..., None]
            count = mask.sum(1, keepdim=True).clamp_min(1)
            active = torch.where(mask, x, 0.0)
            mean = active.sum(1, keepdim=True) / count
            centered = torch.where(mask, x - mean, 0.0)
            covariance = centered.transpose(-1, -2) @ centered / count
            eigenvalues, vectors = torch.linalg.eigh(covariance)
            eigenvalues = eigenvalues.clamp_min(0)
            scale = eigenvalues.mean(-1, keepdim=True) + self.ridge
            rates = 1 + self.strength * eigenvalues / (eigenvalues + scale)
        return PCABasis(vectors, eigenvalues, rates)

    @staticmethod
    def _parameters(coefficients, basis):
        alpha = coefficients.alpha.float()
        sigma = coefficients.sigma.float()
        if alpha.shape != basis.rates.shape[:1] or sigma.shape != alpha.shape:
            raise ValueError("PCA schedule must supply one alpha/sigma per example")
        if not torch.isfinite(alpha).all() or not torch.isfinite(sigma).all():
            raise ValueError("PCA coefficients must be finite")
        if torch.any((alpha < 0) | (alpha > 1) | (sigma < 0)):
            raise ValueError("PCA requires alpha in [0, 1] and sigma >= 0")
        return alpha[:, None].pow(basis.rates), sigma[:, None, None]

    def corrupt(self, clean, coefficients, corruption_mask, *, noise=None):
        basis = self.basis(clean, corruption_mask)
        noise = torch.randn_like(clean) if noise is None else noise
        if noise.shape != clean.shape:
            raise ValueError("corruption noise must match clean state")
        with torch.autocast(clean.device.type, enabled=False):
            attenuation, sigma = self._parameters(coefficients, basis)
            mean = (
                (clean.float() @ basis.vectors) * attenuation[:, None]
            ) @ basis.vectors.transpose(-1, -2)
            # Exact clean endpoint, without round-trip eigensolver error.
            mean = torch.where(
                (coefficients.alpha == 1)[:, None, None], clean.float(), mean
            )
            noisy = mean + sigma * noise.float()
        return torch.where(
            corruption_mask.bool()[..., None], noisy.to(clean.dtype), clean
        )

    def corrupt_state(self, clean, coefficients, corruption_mask):
        return self.corrupt(clean, coefficients, corruption_mask)

    def prior_state(self, clean, coefficients):
        if not torch.all(coefficients.alpha == 0) or not torch.all(
            coefficients.sigma == 1
        ):
            raise ValueError("PCA prior requires the exact alpha=0, sigma=1 endpoint")
        return torch.randn_like(clean)

    def forward_transition(
        self, state, clean, source, target, corruption_mask, *, noise=None
    ):
        """Exact increasing-noise transition CONDITIONAL on the same clean x0.

        This does not define an x0-independent Markov kernel or reverse sampler.
        For s < t, gains a(t)/a(s) and variances sigma(t)^2-gain^2*sigma(s)^2
        reproduce the forward marginals. The source must precede the noise endpoint.
        """
        basis = self.basis(clean, corruption_mask)
        noise = torch.randn_like(state) if noise is None else noise
        if state.shape != clean.shape or noise.shape != state.shape:
            raise ValueError("transition states and noise must match clean shape")
        with torch.autocast(state.device.type, enabled=False):
            a_s, sigma_s = self._parameters(source, basis)
            a_t, sigma_t = self._parameters(target, basis)
            if torch.any(a_s <= 0):
                raise ValueError("source must precede the pure-noise endpoint")
            ratio = a_t / a_s
            variance = sigma_t.square() - ratio[:, None].square() * sigma_s.square()
            if torch.any(variance < -1e-6):
                raise ValueError("inconsistent forward transition: negative variance")
            modes = state.float() @ basis.vectors
            next_modes = (
                ratio[:, None] * modes + variance.clamp_min(0).sqrt() * noise.float()
            )
            result = next_modes @ basis.vectors.transpose(-1, -2)
        return torch.where(
            corruption_mask.bool()[..., None], result.to(state.dtype), state
        )
