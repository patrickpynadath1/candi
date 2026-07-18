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



def test_distillation_trajectory_uses_shared_noise(monkeypatch):
    model = _bare(Distillation, vocab_size=3)
    monkeypatch.setattr(Distillation, "device", property(lambda _self: torch.device("cpu")), raising=False)
    monkeypatch.setattr(Distillation, "dtype", property(lambda _self: torch.float32))
    tokens = torch.tensor([[0, 2]])
    torch.manual_seed(5)
    xt, xs = model._sample_trajectory(
        tokens, torch.tensor([0.0]), torch.tensor([0.0])
    )
    assert torch.equal(xt, xs)
    assert xt.shape == (1, 2, 3)


@pytest.mark.parametrize(
    ("linear", "step", "expected"),
    [(False, 0, 0.125), (False, 2, 0.25), (True, 5, 0.3)],
)
def test_distillation_dt_schedule(linear, step, expected, monkeypatch):
    model = _bare(
        Distillation,
        linear_growth_dt=linear,
        linear_growth_min=0.1,
        linear_growth_max=0.5,
        update_teacher_every=2,
        T=8,
    )
    model._test_step = step
    model._test_trainer = SimpleNamespace(max_steps=10)
    monkeypatch.setattr(Distillation, "global_step", property(lambda self: self._test_step), raising=False)
    monkeypatch.setattr(Distillation, "trainer", property(lambda self: self._test_trainer))
    assert model._compute_dt() == pytest.approx(expected)

