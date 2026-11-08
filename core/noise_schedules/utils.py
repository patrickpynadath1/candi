"""Rank, corruption probability and interpolation utilities for noise schedules."""

import torch


def check_unit_time(time, message="time must be finite and in [0, 1]"):
    """Raise ``ValueError(message)`` unless every time is finite and in [0, 1]."""
    if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError(message)


def build_argmax_error_table(
    vocab_size, sigma_min, sigma_max, num_points, quadrature_order
):
    """One-hot Gaussian argmax error, integrating over the true coordinate.

    P(correct) = E_Z[Phi(Z + 1/sigma)**(V-1)], Z ~ N(0,1).
    Float64 Gauss-Hermite quadrature avoids truncating the Gaussian tails.
    """
    import numpy as np
    from scipy.special import log_ndtr, roots_hermitenorm

    nodes, weights = roots_hermitenorm(quadrature_order)
    weights /= np.sqrt(2 * np.pi)
    sigmas = np.geomspace(sigma_min, sigma_max, num_points)
    errors = []
    for batch in np.array_split(sigmas, max(1, (num_points + 1023) // 1024)):
        log_correct = (vocab_size - 1) * log_ndtr(nodes[None, :] + 1 / batch[:, None])
        # Integrate error directly: 1-E[p] loses precision near zero error.
        errors.append((-np.expm1(log_correct)) @ weights)
    return torch.from_numpy(sigmas), torch.from_numpy(np.concatenate(errors))


def sigma_from_time_vectorized(
    t: torch.Tensor, sigmas: torch.Tensor, errors: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized version of sigma_from_time using linear interpolation.

    Args:
        t: tensor of target monotone noise coordinates (rank fractions or error rates)
        sigmas: 1D tensor of σ values
        errors: 1D tensor of corresponding ε(σ)

    Returns:
        Tensor of interpolated σ values
    """
    t = t.clamp(min=errors[0].item(), max=errors[-1].item())
    t = t.to(sigmas.device)

    # Normalize ε ∈ [0, 1] to indices in [0, len(errors)-1]
    indices = torch.searchsorted(errors, t, right=True).clamp(1, len(errors) - 1)
    i0 = indices - 1
    i1 = indices

    e0 = errors[i0]
    e1 = errors[i1]
    s0 = sigmas[i0]
    s1 = sigmas[i1]

    interp_t = (t - e0) / (e1 - e0 + 1e-8)
    return s0 + interp_t * (s1 - s0)


def expected_rank(d: int, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Normalized expected count of coordinates outranking the true coordinate in a one-hot vector
    corrupted by Gaussian noise.

    Args:
        d: dimension (number of coordinates)
        a: amplitude(s) of the true coordinate, scalar or tensor
        sigma: noise standard deviation(s), scalar or tensor

    Returns:
        Expected outranking count divided by d, same broadcasted shape as a and sigma
    """
    # ensure tensors
    a = torch.as_tensor(a, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)

    # standard normal CDF in torch
    Phi = torch.distributions.Normal(0.0, 1.0).cdf

    # compute p = P(other beats true coordinate)
    p = Phi(-a / (sigma * torch.sqrt(torch.tensor(2.0, dtype=torch.float32))))

    return (d - 1) * p / d
