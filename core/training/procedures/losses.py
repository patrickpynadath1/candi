"""Numerical losses only: no model calls, rollouts, or optimizer side effects."""

import math
import torch
from torch import nn


class TokenCrossEntropy(nn.Module):
    def forward(self, prediction, target, mask):
        return -prediction.float().gather(-1, target[..., None]).squeeze(-1)


class ReverseKL(nn.Module):
    def forward(self, prediction, target, mask):
        p, q = prediction.float(), target.float()
        # Zero-probability classes (e.g. MASK) must not create 0 * inf NaNs.
        delta = torch.where(p.isfinite(), p - q, 0)
        return (p.exp() * delta).sum(-1)


class SequenceAlpha(nn.Module):
    """Mask-normalized sequence alpha loss, with alpha_eff = k / mask_count."""

    def __init__(self, k=0.5):
        super().__init__()
        if not math.isfinite(k) or k < 0:
            raise ValueError("k must be finite and nonnegative")
        self.k = float(k)

    def forward(self, prediction, target, mask):
        logp = prediction.float().gather(-1, target[..., None]).squeeze(-1)
        count = mask.sum(-1)
        mean = torch.where(mask.bool(), logp, 0).sum(-1) / count.clamp_min(1)
        if self.k == 0:
            value = -mean
        else:
            exponent = self.k * mean
            # expm1 is accurate near zero, but its backward uses expm1(x)+1:
            # that rounds to zero for strongly negative FP32 inputs. Use exp
            # in that region to retain the analytically nonzero gradient.
            value = (
                torch.where(exponent > -1, -torch.expm1(exponent), 1 - exponent.exp())
                / self.k
            )
        return torch.where(count > 0, value, 0)


class MeanSquaredError(nn.Module):
    def forward(self, prediction, target, mask):
        return (prediction.float() - target.float()).square().mean(-1)


class MaskedMean(nn.Module):
    def forward(self, values, mask):
        if values.shape != mask.shape:
            raise ValueError("token reduction requires token-shaped values")
        count = mask.sum().to(values)
        total = torch.where(mask.bool(), values, 0).sum()
        return total / count.clamp_min(1), total.detach(), count.detach()


class ExampleMean(nn.Module):
    def forward(self, values, mask):
        if values.shape != mask.shape[:1]:
            raise ValueError("example reduction requires one value per example")
        # Empty examples contribute zero, as specified by the sequence loss.
        count = values.new_tensor(len(values))
        total = values.sum()
        return total / count.clamp_min(1), total.detach(), count.detach()


class DotProduct(nn.Module):
    """Surrogate with a target gradient supplied by a separate estimator."""

    def forward(self, prediction, target, mask):
        return (prediction.float() * target.float()).sum(-1)


class SoftCrossEntropy(nn.Module):
    def forward(self, prediction, target, mask):
        return -(target * torch.where(target > 0, prediction.float(), 0)).sum(-1)
