"""Tabulated von Mises–Fisher path from Chemseddine et al., §3 / Appendix C.

Tables are deterministic numerical quadrature, not learned model parameters.
The two-sided flux evaluation avoids cancellation in the radial tails.
"""

from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
from .spherical import sphere_normalize


@lru_cache(maxsize=8)
def _tables(dimension, kappa_max, resolution, concentrations):
    s = np.linspace(-1, 1, resolution)
    interior = s[1:-1]
    velocities, quantiles = [], []
    u = np.linspace(0, 1, resolution)
    for kappa in np.linspace(0, kappa_max, concentrations):
        logf = (dimension - 3) / 2 * np.log1p(-(interior**2)) + kappa * interior
        density = np.exp(logf - logf.max())
        density = np.concatenate(([0.0], density, [0.0]))
        cdf = cumulative_trapezoid(density, s, initial=0)
        cdf /= cdf[-1]
        mean = np.trapz(s * density, s) / np.trapz(density, s)

        # Integrate positive flux from the nearer boundary in log space.
        def cumulative(log_integrand):
            segments = np.logaddexp(log_integrand[:-1], log_integrand[1:])
            segments += np.log((s[1] - s[0]) / 2)
            return np.concatenate(([-np.inf], np.logaddexp.accumulate(segments)))

        lf = np.concatenate(([-np.inf], logf, [-np.inf]))
        with np.errstate(divide="ignore", invalid="ignore"):
            lower = cumulative(lf + np.log(np.maximum(mean - s, 0)))
            upper = cumulative((lf + np.log(np.maximum(s - mean, 0)))[::-1])[::-1]
            flux = np.where(interior < mean, lower[1:-1], upper[1:-1])
            psi = np.exp(flux - logf - np.log1p(-(interior**2)))
        psi = np.concatenate(
            ([(1 + mean) / (dimension - 1)], psi, [(1 - mean) / (dimension - 1)])
        )
        if kappa == 0:
            psi[:] = 1 / (dimension - 1)
        velocities.append(psi)
        quantiles.append(np.interp(u, cdf, s))
    return torch.tensor(np.stack((velocities, quantiles)), dtype=torch.float32)


class VMFProcess:
    state_geometry = "sphere"

    def __init__(
        self, dimension=512, kappa_max=250.0, resolution=4097, concentrations=257
    ):
        if dimension < 3 or kappa_max <= 0 or resolution < 129 or concentrations < 2:
            raise ValueError(
                "vMF requires dimension >= 3, positive concentration and adequate grids"
            )
        self.dimension, self.kappa_max = dimension, float(kappa_max)
        self.resolution, self.concentrations = resolution, concentrations
        self._device_tables = {}

    def _lookup(self, coordinate, kappa, table):
        if torch.any((kappa < 0) | (kappa > self.kappa_max)):
            raise ValueError("concentration outside vMF table range")
        device = coordinate.device
        if device not in self._device_tables:
            self._device_tables[device] = _tables(
                self.dimension, self.kappa_max, self.resolution, self.concentrations
            ).to(device)
        values = self._device_tables[device][table]
        while kappa.ndim < coordinate.ndim:
            kappa = kappa.unsqueeze(-1)
        x = coordinate.clamp(0, 1) * (self.resolution - 1)
        y = kappa * ((self.concentrations - 1) / self.kappa_max)
        i = x.long().clamp(max=self.resolution - 2)
        j = y.long().clamp(max=self.concentrations - 2)
        a, b = x - i, y - j
        return (1 - b) * ((1 - a) * values[j, i] + a * values[j, i + 1]) + b * (
            (1 - a) * values[j + 1, i] + a * values[j + 1, i + 1]
        )

    def prior_state(self, clean, coefficients):
        return sphere_normalize(torch.randn_like(clean))

    def corrupt_state(self, clean, coefficients, corruption_mask):
        if clean.shape[-1] != self.dimension:
            raise ValueError("vMF dimension must match the representation")
        mean = sphere_normalize(clean.float())
        cosine = self._lookup(torch.rand_like(mean[..., 0]), coefficients[0], 1)
        noise = torch.randn_like(mean)
        tangent = sphere_normalize(noise - (noise * mean).sum(-1, keepdim=True) * mean)
        state = (
            cosine[..., None] * mean
            + (1 - cosine.square()).clamp_min(0).sqrt()[..., None] * tangent
        )
        return torch.where(
            corruption_mask.bool()[..., None], state.to(clean.dtype), clean
        )

    def velocity(self, state, codebook, probabilities, kappa):
        x, w = sphere_normalize(state.float()), sphere_normalize(codebook.float())
        cosine = (x @ w.T).clamp(-1, 1)
        weights = probabilities.float() * self._lookup((cosine + 1) / 2, kappa, 0)
        mean = weights @ w
        return mean - (weights * cosine).sum(-1, keepdim=True) * x

    def score(self, state, codebook, probabilities, kappa):
        x = sphere_normalize(state.float())
        mean = probabilities.float() @ sphere_normalize(codebook.float())
        while kappa.ndim < x.ndim:
            kappa = kappa.unsqueeze(-1)
        return kappa * (mean - (mean * x).sum(-1, keepdim=True) * x)
