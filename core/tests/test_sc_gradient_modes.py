"""Check producer/consumer parameter gradients independently."""

import copy
import pytest
import torch
from torch import nn
from core.models.prediction import ModelPrediction
from core.modifiers.self_conditioning.prediction import PredictionCoordinator


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.producer = nn.Parameter(torch.tensor(0.7))
        self.consumer = nn.Parameter(torch.tensor(1.3))

    def forward(self, x, self_condition=None, return_features=False):
        if self_condition is None:
            logits = torch.stack((x * self.producer, -x * self.producer), -1)
        else:
            logits = self_condition * self.consumer
        return ModelPrediction(logits, logits) if return_features else logits


class Predictor:
    def training_posterior(self, logits, observed):
        return logits

    def probabilities(self, logits):
        return logits.softmax(-1)


@pytest.mark.parametrize("mode", ["detached", "both", "producer", "mixed"])
def test_gradient_routing(mode):
    model = Model()
    reference = copy.deepcopy(model)
    x = torch.arange(1.0, 5.0).reshape(4, 1)
    observed = torch.zeros(4, 1, dtype=torch.long)
    policy = PredictionCoordinator(enabled=True, probability=1, gradient_mode=mode)
    torch.manual_seed(7)
    actual = policy.predict(
        model, Predictor(), (x,), observed, "training_posterior", mean=lambda p: p
    )
    actual[..., 0].sum().backward()
    probabilities = reference(x, return_features=True).logits.softmax(-1)
    if mode == "mixed":
        torch.manual_seed(7)
        producer = torch.zeros(4, dtype=torch.bool)
        producer[torch.randperm(4)[:2]] = True
    else:
        producer = torch.full((4,), mode == "producer")
    if mode == "detached":
        expected = probabilities.detach() * reference.consumer
    elif mode == "both":
        expected = probabilities * reference.consumer
    else:
        expected = torch.where(
            producer[:, None, None],
            probabilities * reference.consumer.detach(),
            probabilities.detach() * reference.consumer,
        )
    expected[..., 0].sum().backward()
    torch.testing.assert_close(actual, expected)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(
            torch.zeros_like(p) if p.grad is None else p.grad,
            torch.zeros_like(q) if q.grad is None else q.grad,
        )


def test_identity_conditioning_preserves_feedback_and_input_gradients():
    from core.models.prediction import ConditioningProjection

    adapter = ConditioningProjection(4, 4, initialization="identity")
    hidden = torch.randn(2, 3, 4)
    feedback = torch.randn_like(hidden, requires_grad=True)
    torch.testing.assert_close(adapter(hidden, feedback), hidden + feedback)
    adapter(hidden, feedback).sum().backward()
    torch.testing.assert_close(feedback.grad, torch.ones_like(feedback))


@pytest.mark.parametrize("mode", ["producer", "mixed"])
def test_frozen_consumer_restores_shared_modules_and_tied_parameters(mode):
    from core.training.ema import ExponentialMovingAverage

    class SharedModel(Model):
        def __init__(self):
            super().__init__()
            self.mapping = nn.Linear(2, 2)
            self.alias = self.mapping
            self.readout = nn.Linear(2, 2, bias=False)
            self.readout.weight = self.mapping.weight

        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            logits = self.readout(self.alias(output.logits))
            return ModelPrediction(logits, logits)

    model = SharedModel()
    original = dict(model.named_parameters(remove_duplicate=False))
    ema = ExponentialMovingAverage(model.parameters(), 0.99)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    policy = PredictionCoordinator(enabled=True, probability=1, gradient_mode=mode)
    x = torch.arange(1.0, 5.0).reshape(4, 1)
    observed = torch.zeros(4, 1, dtype=torch.long)
    for _ in range(3):
        output = policy.predict(
            model, Predictor(), (x,), observed, "training_posterior", mean=lambda p: p
        )
        output.square().sum().backward()
        current = dict(model.named_parameters(remove_duplicate=False))
        assert current.keys() == original.keys()
        assert all(current[n] is p and p.requires_grad for n, p in original.items())
        optimizer.step()
        optimizer.zero_grad()
        ema.update(model.parameters())
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        ema.restore(model.parameters())
    assert model.readout.weight is model.mapping.weight
    assert model.alias is model.mapping
