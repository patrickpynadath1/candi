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


@pytest.mark.parametrize("teacher_ema", [False, True])
def test_distillation_teacher_update_uses_configured_student_source(
    teacher_ema, monkeypatch
):
    model = _bare(
        Distillation,
        backbone=torch.nn.Linear(2, 2, bias=False),
        teacher=torch.nn.Linear(2, 2, bias=False),
        teacher_ema=teacher_ema,
        update_teacher_every=2,
    )
    model._test_step = 4
    monkeypatch.setattr(
        Distillation,
        "global_step",
        property(lambda self: self._test_step),
        raising=False,
    )
    model.backbone.weight.data.fill_(3.0)
    model.teacher.weight.data.zero_()

    if teacher_ema:
        model.ema = SimpleNamespace(
            copy_to=lambda parameters: [parameter.data.fill_(7.0) for parameter in parameters]
        )
    model._maybe_update_teacher_weights()

    expected = 7.0 if teacher_ema else 3.0
    assert torch.equal(model.teacher.weight, torch.full_like(model.teacher.weight, expected))


def test_distillation_teacher_is_not_updated_between_refresh_steps(monkeypatch):
    model = _bare(
        Distillation,
        backbone=torch.nn.Linear(2, 2, bias=False),
        teacher=torch.nn.Linear(2, 2, bias=False),
        teacher_ema=False,
        update_teacher_every=3,
    )
    model._test_step = 2
    monkeypatch.setattr(
        Distillation,
        "global_step",
        property(lambda self: self._test_step),
        raising=False,
    )
    model.backbone.weight.data.fill_(5.0)
    original = model.teacher.weight.detach().clone()
    model._maybe_update_teacher_weights()
    assert torch.equal(model.teacher.weight, original)
