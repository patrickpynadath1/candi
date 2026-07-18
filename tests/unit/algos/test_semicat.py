# ruff: noqa: F401
from types import SimpleNamespace

import pytest
import torch

from algos.cont_diff_embed import ContDiffEmbed
from algos.cont_diff_onehot import ContDiffOnehot
from algos.d3pm_absorb import D3PMAbsorb
from algos.distillation import Distillation
from algos.sedd_absorb import SEDDAbsorb
from algos.semicat import Semicat, _view_for


def _bare(cls, **attrs):
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    model._device = torch.device("cpu")
    model._dtype = torch.float32
    for key, value in attrs.items():
        setattr(model, key, value)
    return model



def test_semicat_interpolation_endpoints():
    noise = torch.randn(2, 3, 4)
    tokens = torch.tensor([[0, 1, 2], [3, 2, 1]])
    at_zero = Semicat._interpolate(noise, tokens, torch.zeros(2, 1, 1))
    at_one = Semicat._interpolate(noise, tokens, torch.ones(2, 1, 1))
    assert torch.equal(at_zero, noise)
    assert torch.equal(at_one, torch.nn.functional.one_hot(tokens, 4).float())


def test_semicat_view_for_broadcasts_batch_values():
    result = _view_for(torch.tensor([0.2, 0.8]), torch.zeros(2, 3, 4))
    assert result.shape == (2, 1, 1)


def test_semicat_nll_shape_finite_and_backward():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, _sigma):
            return self.linear(x)

    model = _bare(Semicat, backbone=Backbone(), vocab_size=4, label_smoothing=0.1)
    loss = model.nll(torch.tensor([[0, 1, 2], [3, 2, 1]]), None)
    assert loss.shape == (2, 3)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert torch.isfinite(model.backbone.linear.weight.grad).all()


def test_semicat_sampling_rejects_unknown_method():
    model = _bare(
        Semicat, vocab_size=3, num_tokens=2,
        config=SimpleNamespace(sampling=SimpleNamespace(steps=1)),
        sampling_method="unknown",
    )
    with pytest.raises(ValueError, match="Unknown sampling method"):
        model.generate_samples(1)


def test_semicat_sampling_preserves_prompt(monkeypatch):
    model = _bare(
        Semicat, vocab_size=4, num_tokens=3,
        config=SimpleNamespace(sampling=SimpleNamespace(steps=1)),
        sampling_method="flow_map",
    )
    monkeypatch.setattr(Semicat, "device", property(lambda _self: torch.device("cpu")), raising=False)
    model.prior_sample = lambda *_shape: torch.zeros(1, 3, 4)
    model.sample_semicat_flow_map = lambda x, *_args: x
    prompt = torch.tensor([[2, 1, 3]])
    mask = torch.tensor([[1, 0, 1]])
    result = model.generate_samples(1, prompt_tokens=prompt, prompt_mask=mask)
    assert torch.equal(result[mask.bool()], prompt[mask.bool()])
