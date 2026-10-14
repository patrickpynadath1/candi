"""Training-objective math, gradients, and checkpoint ownership."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from core.objectives.cross_entropy import CrossEntropy
from core.objectives.weighting import (
    LearnedLossWeighting,
    ObjectiveContext,
    PowerLossWeighting,
)


def context():
    return ObjectiveContext(
        time=torch.tensor([0.2, 0.7]),
        alpha=torch.tensor([0.8, 0.3]),
        sigma=torch.tensor([0.2, 0.7]),
    )


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 1.0])
def test_smoothing_matches_torch_value_and_gradient(smoothing):
    torch.manual_seed(41)
    logits = torch.randn(2, 3, 7, requires_grad=True)
    tokens = torch.randint(7, (2, 3))
    actual = CrossEntropy(smoothing)(logits.log_softmax(-1), tokens)
    expected = F.cross_entropy(
        logits.transpose(1, 2), tokens, reduction="none", label_smoothing=smoothing
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), logits)[0],
    )


def test_smoothing_rejects_excluded_classes_but_default_preserves_them():
    log_probs = torch.tensor([[[0.0, -torch.inf]]])
    targets = torch.zeros(1, 1, dtype=torch.long)
    assert CrossEntropy()(log_probs, targets).item() == 0
    with pytest.raises(ValueError, match="excluded tokens"):
        CrossEntropy(0.1)(log_probs, targets)
    for value in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError):
            CrossEntropy(value)


def test_power_weighting_and_context_contract():
    loss = torch.ones(2, 3)
    weighting = PowerLossWeighting("sigma", exponent=-2, scale=0.5)
    torch.testing.assert_close(
        weighting(loss, context()), loss * (0.5 / context().sigma.square())[:, None]
    )
    assert not CrossEntropy().requires_context
    objective = CrossEntropy(weighting=weighting)
    assert objective.requires_context
    with pytest.raises(ValueError, match="requires ObjectiveContext"):
        objective(torch.zeros(2, 3, 1), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="shape"):
        weighting(
            loss,
            ObjectiveContext(torch.zeros(2), torch.ones(2), torch.ones(2, 1)),
        )
    with pytest.raises(ValueError, match="batch"):
        weighting(torch.ones(3, 2), context())


def test_learned_weighting_neutral_initialization_formula_and_registration():
    weighting = LearnedLossWeighting(cond_dim=8, hidden_dim=12)
    objective = CrossEntropy(weighting=weighting)
    owner = nn.Module()
    owner.objective = objective
    logits = torch.randn(2, 3, 7, requires_grad=True)
    tokens = torch.randint(7, (2, 3))
    ce = CrossEntropy()(logits.log_softmax(-1), tokens)
    torch.testing.assert_close(
        objective(logits.log_softmax(-1), tokens, context=context()), ce
    )
    with torch.no_grad():
        weighting.mlp[-1].bias.fill_(0.4)
    actual = objective(logits.log_softmax(-1), tokens, context=context())
    torch.testing.assert_close(actual, math.exp(-0.4) * ce + 0.4)
    # Reduction masks the additive term as well as the weighted CE.
    mask = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    (actual * mask).sum().backward()
    torch.testing.assert_close(
        weighting.mlp[-1].bias.grad.squeeze(),
        ((1 - math.exp(-0.4) * ce.detach()) * mask).sum(),
    )
    assert weighting.mlp[-1].weight.grad.abs().sum() > 0
    assert "objective.weighting.mlp.2.bias" in owner.state_dict()
    optimizer = torch.optim.SGD(owner.parameters(), lr=0.1)
    optimizer.step()
    restored = CrossEntropy(weighting=LearnedLossWeighting(cond_dim=8, hidden_dim=12))
    restored.load_state_dict(objective.state_dict())
    torch.testing.assert_close(
        restored(logits.log_softmax(-1), tokens, context=context()),
        objective(logits.log_softmax(-1), tokens, context=context()),
    )


@pytest.mark.parametrize(
    "name", ["time", "alpha", "sigma", "effective_noise", "snr", "log_snr"]
)
def test_conditioning_endpoint_values_are_finite(name):
    endpoints = ObjectiveContext(
        torch.tensor([0.0, 1]), torch.tensor([1.0, 0]), torch.tensor([0.0, 1])
    )
    result = LearnedLossWeighting(name, cond_dim=8, hidden_dim=12)(
        torch.ones(2, 3), endpoints
    )
    torch.testing.assert_close(result, torch.ones(2, 3))


def test_time_weighting_only_accesses_time():
    context = ObjectiveContext(time=torch.tensor([0.2, 0.7]))
    loss = torch.ones(2, 3)
    torch.testing.assert_close(
        PowerLossWeighting("time", exponent=2)(loss, context),
        loss * context.time.square()[:, None],
    )
    torch.testing.assert_close(
        LearnedLossWeighting("time", cond_dim=8, hidden_dim=12)(loss, context), loss
    )


def test_weighting_only_validates_accessed_coordinates():
    loss = torch.ones(2, 3)
    context = ObjectiveContext(time=torch.zeros(2, 1), alpha=torch.tensor([0.8, 0.3]))
    torch.testing.assert_close(
        PowerLossWeighting("alpha", exponent=1)(loss, context),
        loss * context.alpha[:, None],
    )
    with pytest.raises(ValueError, match="context.sigma"):
        PowerLossWeighting("sigma")(loss, context)
    with pytest.raises(ValueError, match="context.alpha and context.sigma"):
        LearnedLossWeighting("snr", cond_dim=8, hidden_dim=12)(loss, context)


def test_coordinate_gradients_are_not_detached():
    sigma = torch.tensor([0.2, 0.7], requires_grad=True)
    coordinates = ObjectiveContext(torch.tensor([0.2, 0.7]), torch.ones(2), sigma)
    loss = PowerLossWeighting("sigma", exponent=2)(torch.ones(2, 3), coordinates)
    loss.sum().backward()
    torch.testing.assert_close(sigma.grad, 6 * sigma.detach())
