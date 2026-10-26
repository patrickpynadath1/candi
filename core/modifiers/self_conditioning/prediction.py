"""Prediction policy; the caller supplies the model, predictor, and mean conversion."""

from dataclasses import dataclass
from dataclasses import fields, is_dataclass, replace
from contextlib import nullcontext
import math

import torch

from core.models.prediction import ModelPrediction
from .sources import final_hidden, SOURCES


@dataclass
class TrajectoryState:
    conditioning: torch.Tensor | None = None
    pending: torch.Tensor | None = None
    staged: bool = False

    def update(self, value):
        if self.staged:
            self.pending = value
        else:
            self.conditioning = value


class PredictionCoordinator:
    def __init__(
        self,
        enabled=False,
        source="posterior_mean",
        probability=0.9,
        evaluation_probability=1.0,
        selection="bernoulli",
        gradient_mode="detached",
    ):
        if source not in SOURCES:
            raise ValueError(
                "self-conditioning source must be posterior_mean or hidden"
            )
        if not 0 <= probability <= 1 or not 0 <= evaluation_probability <= 1:
            raise ValueError("self-conditioning probability must be in [0, 1]")
        self.enabled, self.source, self.probability = enabled, source, probability
        self.evaluation_probability = evaluation_probability
        if selection not in {"bernoulli", "fixed_count"}:
            raise ValueError("selection must be bernoulli or fixed_count")
        self.selection = selection
        if gradient_mode not in {"detached", "both", "producer", "mixed"}:
            raise ValueError("unknown self-conditioning gradient mode")
        self.gradient_mode = gradient_mode

    def select(self, batch, device, training):
        probability = self.probability if training else self.evaluation_probability
        if not self.enabled or probability == 0:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        if self.selection == "fixed_count":
            selected = torch.zeros(batch, dtype=torch.bool, device=device)
            selected[
                torch.randperm(batch, device=device)[: math.ceil(batch * probability)]
            ] = True
            return selected
        return (
            torch.ones(batch, dtype=torch.bool, device=device)
            if probability == 1
            else torch.rand(batch, device=device) < probability
        )

    def validate(self, model, predictor, training_pathway="training_posterior"):
        architectural = getattr(model, "conditioning_adapter", None) is not None
        if self.enabled != architectural:
            raise ValueError(
                "prediction policy and model must enable self-conditioning together"
            )
        if (
            self.enabled
            and self.source == "posterior_mean"
            and not (
                callable(getattr(predictor, training_pathway, None))
                and callable(getattr(predictor, "probabilities", None))
            )
        ):
            raise TypeError(
                f"self-conditioning source={self.source!r} requires clean-token probabilities; "
                f"{type(predictor).__name__} does not expose {training_pathway} and probabilities. "
                "Use source=hidden where hidden-state conditioning is integrated, or provide an explicit clean-token probability conversion."
            )

    def feedback(
        self,
        output,
        predictor,
        observed,
        mean,
        logits_transform,
        training_pathway="training_posterior",
        detach=True,
    ):
        if not isinstance(output, ModelPrediction):
            raise TypeError(
                "self-conditioning requires logits and hidden model outputs"
            )
        with torch.no_grad() if detach else nullcontext():
            if self.source == "hidden":
                return (
                    final_hidden(output.hidden).detach()
                    if detach
                    else final_hidden(output.hidden)
                )
            probabilities = predictor.probabilities(
                getattr(predictor, training_pathway)(
                    logits_transform(output.logits), observed
                )
            )
            value = mean(probabilities)
            return value.detach() if detach else value

    def predict(
        self,
        model,
        predictor,
        inputs,
        observed,
        pathway,
        mean=None,
        logits_transform=lambda value: value,
        trajectory=None,
        selected=None,
        bootstrap_logits_transform=None,
        bootstrap_feedback_transform=lambda value: value,
        training_pathway="training_posterior",
    ):
        parameterize = getattr(predictor, pathway)
        if not self.enabled:
            return parameterize(logits_transform(model(*inputs)), observed)
        conditioning = None
        mode = (
            self.gradient_mode
            if model.training and torch.is_grad_enabled() and trajectory is None
            else "detached"
        )
        producer = None
        probability = (
            self.probability if model.training else self.evaluation_probability
        )
        if trajectory is not None:
            conditioning = trajectory.conditioning
        elif pathway == training_pathway and (selected is not None or probability > 0):
            batch = len(observed)
            if selected is None:
                selected = self.select(batch, observed.device, model.training)
            if selected.any():
                with (
                    torch.no_grad() if mode == "detached" else nullcontext(),
                    torch.autocast(
                        observed.device.type,
                        enabled=torch.is_autocast_enabled(observed.device.type),
                        cache_enabled=False,
                    ),
                ):
                    context = (
                        model.select_context(selected)
                        if hasattr(model, "select_context")
                        else nullcontext()
                    )
                    with context:
                        bootstrap = model(
                            *(_select_batch(value, selected) for value in inputs),
                            return_features=True,
                        )
                    feedback = self.feedback(
                        bootstrap,
                        predictor,
                        observed[selected],
                        mean,
                        bootstrap_logits_transform or logits_transform,
                        training_pathway,
                        detach=mode == "detached",
                    )
                    feedback = bootstrap_feedback_transform(feedback)
                    conditioning = feedback.new_zeros((batch, *feedback.shape[1:]))
                    conditioning[selected] = feedback
        if mode in {"producer", "mixed"} and conditioning is not None:
            producer = selected.clone()
            if mode == "mixed":
                indices = selected.nonzero().flatten()
                producer.zero_()
                indices = indices[torch.randperm(len(indices), device=indices.device)]
                producer[indices[: len(indices) // 2]] = True
            shape = (len(producer),) + (1,) * (conditioning.ndim - 1)
            conditioning = torch.where(
                producer.reshape(shape), conditioning, conditioning.detach()
            )
            # Detached parameter values freeze the consumer while preserving its
            # Jacobian with respect to the producer's conditioning tensor.
            with torch.autocast(
                observed.device.type,
                enabled=torch.is_autocast_enabled(observed.device.type),
                cache_enabled=False,
            ):
                frozen = torch.func.functional_call(
                    model,
                    _detached_parameters(model),
                    inputs,
                    dict(self_condition=conditioning, return_features=True),
                    tie_weights=False,
                )
                output = model(
                    *inputs, self_condition=conditioning.detach(), return_features=True
                )
            logits = torch.where(producer[:, None, None], frozen.logits, output.logits)
            return parameterize(logits_transform(logits), observed)
        output = model(*inputs, self_condition=conditioning, return_features=True)
        if trajectory is not None:
            trajectory.update(
                self.feedback(
                    output,
                    predictor,
                    observed,
                    mean,
                    logits_transform,
                    training_pathway,
                )
            )
        return parameterize(logits_transform(output.logits), observed)


def _select_batch(value, selected):
    """Select a batch through nested immutable model-input structures."""
    if callable(getattr(value, "select_batch", None)):
        return value.select_batch(selected)
    if torch.is_tensor(value):
        return value[selected]
    if isinstance(value, tuple):
        return type(value)(_select_batch(item, selected) for item in value)
    if isinstance(value, list):
        return [_select_batch(item, selected) for item in value]
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _select_batch(getattr(value, field.name), selected)
                for field in fields(value)
            },
        )
    return value


def _detached_parameters(model):
    """Swap each physical parameter slot once, preserving tied parameter values.

    A shared submodule can have several paths (e.g. a view representation also
    exposed through the backbone). Expanding all aliases in functional_call can
    restore a detached value over the original Parameter on PyTorch 2.5.
    named_modules visits each module once; local parameter names retain distinct
    slots even when their values are tied across modules.
    """
    values = {}
    parameters = {}
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            if id(parameter) not in values:
                values[id(parameter)] = parameter.detach()
            key = f"{module_name}.{name}" if module_name else name
            parameters[key] = values[id(parameter)]
    return parameters
