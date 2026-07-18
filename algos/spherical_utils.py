"""Numerically stable geometry helpers for unit hyperspheres."""

import torch


def sphere_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def slerp(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    alpha_t: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Interpolate from ``clean`` (alpha=0) to ``noisy`` (alpha=1)."""
    clean = sphere_normalize(clean, eps)
    noisy = sphere_normalize(noisy, eps)
    while alpha_t.ndim < clean.ndim:
        alpha_t = alpha_t.unsqueeze(-1)
    cosine = (clean * noisy).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cosine)
    sin_omega = omega.sin().clamp_min(eps)
    out = (
        torch.sin((1 - alpha_t) * omega) / sin_omega * clean
        + torch.sin(alpha_t * omega) / sin_omega * noisy
    )
    return sphere_normalize(out, eps)


def log_map(x: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Map ``target`` to a tangent vector at ``x``."""
    x = sphere_normalize(x, eps)
    target = sphere_normalize(target, eps)
    cosine = (x * target).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cosine)
    direction = target - cosine * x
    return direction * (omega / direction.norm(dim=-1, keepdim=True).clamp_min(eps))


def exp_map(x: torch.Tensor, delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Move from ``x`` along a tangent vector and return to the sphere."""
    x = sphere_normalize(x, eps)
    norm = delta.norm(dim=-1, keepdim=True)
    direction = delta / norm.clamp_min(eps)
    out = torch.cos(norm) * x + torch.sin(norm) * direction
    return sphere_normalize(out, eps)
