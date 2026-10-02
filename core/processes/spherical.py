"""Geometry helpers for unit hyperspheres, including coincident endpoints."""

import math

import torch


def sphere_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def slerp(clean, noisy, alpha_t, eps=1e-6):
    """Interpolate from clean (alpha=0) to noisy (alpha=1).

    Antipodes have no unique shortest arc; use log_map's deterministic arc.
    """
    while alpha_t.ndim < clean.ndim:
        alpha_t = alpha_t.unsqueeze(-1)
    return exp_map(clean, alpha_t * log_map(clean, noisy, eps), eps)


def log_map(x: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Map target into the tangent space at x; log_x(x) is zero."""
    x = sphere_normalize(x, eps)
    target = sphere_normalize(target, eps)
    cosine = (x * target).sum(-1, keepdim=True).clamp(-1, 1)
    tangent = target - cosine * x
    tangent = tangent - (tangent * x).sum(-1, keepdim=True) * x
    sine = tangent.norm(dim=-1, keepdim=True)
    angle = torch.atan2(sine, cosine)
    tiny = torch.finfo(x.dtype).eps
    result = tangent * (angle / sine.clamp_min(tiny))
    # Choose an orthogonal coordinate direction only at the antipodal singularity.
    axis = torch.zeros_like(x).scatter_(-1, x.abs().argmin(-1, keepdim=True), 1)
    orthogonal = sphere_normalize(axis - (axis * x).sum(-1, keepdim=True) * x, eps)
    return torch.where((sine <= tiny) & (cosine < 0), math.pi * orthogonal, result)


def exp_map(x: torch.Tensor, delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Move along a tangent vector, with a smooth zero-length limit."""
    x = sphere_normalize(x, eps)
    norm = delta.norm(dim=-1, keepdim=True)
    out = torch.cos(norm) * x + torch.sinc(norm / math.pi) * delta
    return sphere_normalize(out, eps)


class SphericalProcess:
    def __init__(self, eps=1e-6, precision="float64"):
        if precision not in {"float32", "float64"}:
            raise ValueError("unknown spherical precision")
        self.eps, self.precision = eps, precision

    def prior(self, clean):
        return sphere_normalize(torch.randn_like(clean), self.eps)

    def corrupt(self, clean, keep, corruption_mask):
        noisy = self.prior(clean)
        dtype = torch.float64 if self.precision == "float64" else torch.float32
        state = slerp(
            clean.to(dtype), noisy.to(dtype), (1 - keep).to(dtype), self.eps
        ).to(clean.dtype)
        return torch.where(corruption_mask.bool().unsqueeze(-1), state, clean)

    def advance(self, state, tangent):
        return exp_map(state, tangent, self.eps)
