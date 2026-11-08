"""Coefficients of the Gaussian path x(t)=alpha(t)*x0+sigma(t)*epsilon."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GaussianPath:
    alpha: torch.Tensor
    sigma: torch.Tensor
    dalpha_dt: torch.Tensor
    dsigma_dt: torch.Tensor


def path_from_gamma(gamma, derivative):
    """The VP Gaussian path at log noise-to-signal ``gamma`` and its time derivative.

    alpha**2 = sigmoid(-gamma), sigma**2 = sigmoid(gamma).
    """
    alpha_sq, sigma_sq = torch.sigmoid(-gamma), torch.sigmoid(gamma)
    alpha, sigma = alpha_sq.sqrt(), sigma_sq.sqrt()
    return GaussianPath(
        alpha,
        sigma,
        -0.5 * derivative * sigma_sq * alpha,
        0.5 * derivative * alpha_sq * sigma,
    )
