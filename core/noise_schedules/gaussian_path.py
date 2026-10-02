"""Coefficients of the Gaussian path x(t)=alpha(t)*x0+sigma(t)*epsilon."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GaussianPath:
    alpha: torch.Tensor
    sigma: torch.Tensor
    dalpha_dt: torch.Tensor
    dsigma_dt: torch.Tensor
