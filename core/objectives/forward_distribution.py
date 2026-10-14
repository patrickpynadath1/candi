"""Composable divergences after applying a discrete forward marginal."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ForwardKL(nn.Module):
    """KL(target forward marginal || predicted forward marginal), per token."""

    def forward(self, target, predicted, observed):
        # Zero target mass contributes zero, including at exact endpoints.
        tiny = torch.finfo(predicted.dtype).tiny
        return (
            target * (target.clamp_min(tiny).log() - predicted.clamp_min(tiny).log())
        ).sum(-1)


class ObservedTokenIS(nn.Module):
    """Pointwise Itakura–Saito penalty at the observed noisy token."""

    def forward(self, target, predicted, observed):
        tiny = torch.finfo(predicted.dtype).tiny
        q = target.gather(-1, observed[..., None]).squeeze(-1)
        p = predicted.gather(-1, observed[..., None]).squeeze(-1)
        log_ratio = q.clamp_min(tiny).log() - p.clamp_min(tiny).log()
        return torch.expm1(log_ratio) - log_ratio


class ForwardDistributionObjective(nn.Module):
    """Apply one view's forward marginal once, then sum configured loss terms.

    A process must expose marginal(clean_probabilities, coefficients). Terms
    consume target/predicted probabilities and the observed discrete state.
    Weights here are ablation coefficients, not diffusion ELBO weights.
    """

    requires_context = True

    def __init__(self, terms, weights=None, view_index=0, weighting=None):
        super().__init__()
        if not terms:
            raise ValueError("at least one forward-distribution loss term is required")
        self.terms = nn.ModuleList(terms)
        self.weights = tuple(weights if weights is not None else [1.0] * len(terms))
        if (
            len(self.weights) != len(terms)
            or any(not math.isfinite(w) or w < 0 for w in self.weights)
            or not any(self.weights)
        ):
            raise ValueError(
                "term weights must match terms and be finite, nonnegative, and not all zero"
            )
        if not isinstance(view_index, int) or view_index < 0:
            raise ValueError("view_index must be a nonnegative integer")
        self.view_index = view_index
        self.weighting = weighting

    def forward(self, log_probs, targets, *, context=None):
        if context is None or not (
            len(context.processes) == len(context.coefficients) == len(context.states)
            and self.view_index < len(context.processes)
        ):
            raise ValueError(
                "forward-distribution objective requires a populated view context"
            )
        process = context.processes[self.view_index]
        marginal = getattr(process, "marginal", None)
        if not callable(marginal):
            raise TypeError(
                "selected process must expose marginal(clean_probabilities, coefficients)"
            )
        observed = context.states[self.view_index]
        if observed.shape != targets.shape or observed.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "forward-distribution objective requires a discrete token state"
            )
        coefficients = context.coefficients[self.view_index]
        # Keep probability arithmetic outside reduced precision under autocast.
        probabilities = log_probs.double().exp()
        target = marginal(
            F.one_hot(targets, log_probs.shape[-1]).double(), coefficients
        )
        predicted = marginal(probabilities, coefficients)
        loss = sum(
            weight * term(target, predicted, observed)
            for weight, term in zip(self.weights, self.terms)
            if weight != 0
        )
        return loss if self.weighting is None else self.weighting(loss, context)
