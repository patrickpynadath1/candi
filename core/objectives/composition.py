"""Explicit argument binding and tokenwise composition of objective components."""

import inspect
import math

from torch import nn


class Component(nn.Module):
    def __init__(self, component, inputs, coefficient=1.0):
        super().__init__()
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("component coefficient must be finite and nonnegative")
        inspect.signature(component.forward).bind(**{key: None for key in inputs})
        self.component, self.inputs = component, dict(inputs)
        self.coefficient = coefficient

    def validate(self, available):
        missing = set(self.inputs.values()) - set(available)
        if missing:
            raise ValueError(f"unknown objective inputs: {sorted(missing)}")

    def forward(self, values):
        return self.coefficient * self.component(
            **{argument: values[source] for argument, source in self.inputs.items()}
        )


class SumTerms(nn.Module):
    """Shared combination of explicit components; returns total and raw terms."""

    def __init__(self, terms):
        super().__init__()
        self.terms = nn.ModuleList(terms)

    def forward(self, values):
        raw = [
            term.component(
                **{arg: values[source] for arg, source in term.inputs.items()}
            )
            for term in self.terms
        ]
        return sum(
            term.coefficient * value for term, value in zip(self.terms, raw)
        ), raw


class ComposedObjective(nn.Module):
    requires_context = True

    def __init__(self, terms, preparation=None, weighting=None):
        super().__init__()
        if not terms:
            raise ValueError("objective needs at least one term")
        self.combination = SumTerms(terms)
        self.preparation, self.weighting = preparation, weighting
        available = {"log_probs", "targets"}
        if preparation is not None:
            available.update(preparation.outputs)
        for component in [*self.terms, *([] if weighting is None else [weighting])]:
            component.validate(available)

    @property
    def terms(self):
        return self.combination.terms

    def forward(self, log_probs, targets, *, context=None):
        values = {"log_probs": log_probs, "targets": targets}
        if self.preparation is not None:
            values.update(self.preparation(log_probs, targets, context=context))
        loss, _ = self.combination(values)
        if not hasattr(loss, "shape") or loss.shape != targets.shape:
            raise ValueError("objective terms must produce tokenwise losses")
        if self.weighting is not None:
            weight = self.weighting(values)
            if weight.shape != loss.shape:
                raise ValueError("objective weighting must produce tokenwise weights")
            loss = loss * weight
        return loss
