"""Vocabulary-dependent time maps used by Flow Language Models."""

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.special import log_ndtr


def compute_alpha_exact(gamma, vocab_size: int, n_gh: int = 100):
    """Reference FLM Gauss-Hermite gamma-to-alpha map."""
    tensor = isinstance(gamma, torch.Tensor)
    values = gamma.detach().cpu().numpy() if tensor else np.asarray(gamma)
    sigma = np.maximum(1.0 - values, 1e-12)
    margin = values / sigma
    nodes, weights = hermgauss(n_gh)
    weights = weights / np.sqrt(np.pi)
    log_probability = (vocab_size - 1) * log_ndtr(
        np.sqrt(2.0) * nodes[None, :] + margin.reshape(-1, 1)
    )
    correct = np.sum(weights * np.exp(log_probability), axis=-1)
    alpha = vocab_size / (vocab_size - 1.0) * (correct - 1.0 / vocab_size)
    alpha = np.clip(alpha + (values.reshape(-1) - 1) * 1e-10, 0.0, 1.0)
    alpha = alpha.reshape(values.shape)
    if tensor:
        return torch.as_tensor(alpha, device=gamma.device, dtype=gamma.dtype)
    return alpha


def build_luts(vocab_size: int, size: int = 10_000):
    gamma = torch.linspace(0, 1, size, dtype=torch.float64)
    alpha = compute_alpha_exact(gamma, vocab_size)
    alpha = torch.cummax(alpha, 0).values
    return torch.stack((alpha, gamma)), torch.stack((gamma, alpha))


def _interp(values: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    lut = lut.to(values.device)
    x, y = lut[0], lut[1]
    flat = values.double().flatten().clamp(x[0], x[-1])
    idx = torch.searchsorted(x, flat).clamp(1, x.numel() - 1)
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    out = y0 + (flat - x0) / (x1 - x0).clamp_min(1e-12) * (y1 - y0)
    return out.reshape(values.shape).to(values.dtype)


def alpha_to_gamma(alpha: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return _interp(alpha, lut)


def gamma_to_alpha(gamma: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return _interp(gamma, lut)
