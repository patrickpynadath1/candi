# ruff: noqa: F401
from types import SimpleNamespace

import pytest
import torch

from algos.ar import AR
from algos.cadd import CADD
from algos.candi import CANDI
from algos.duo import DUO, DUO_BASE, Integral
from algos.mdlm import MDLM


def _bare(cls, **attrs):
    instance = object.__new__(cls)
    torch.nn.Module.__init__(instance)
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance



def test_duo_posterior_is_a_probability_distribution():
    model = _bare(
        DUO_BASE,
        vocab_size=3,
        config=SimpleNamespace(sampling=SimpleNamespace(use_float64=False)),
    )
    model._device = torch.device("cpu")
    model._dtype = torch.float32
    x = torch.full((1, 2, 3), 1 / 3)
    xt = torch.tensor([[0, 2]])
    posterior = model._compute_posterior(
        x, xt, torch.tensor([[0.8]]), torch.tensor([[0.5]])
    )
    assert posterior.shape == (1, 2, 3)
    assert torch.isfinite(posterior).all()
    assert torch.all(posterior >= 0)
    assert torch.allclose(posterior.sum(-1), torch.ones(1, 2))


def test_duo_posterior_moves_toward_model_prediction_when_more_noise_is_removed():
    model = _bare(
        DUO_BASE,
        vocab_size=3,
        config=SimpleNamespace(sampling=SimpleNamespace(use_float64=False)),
    )
    model._device = torch.device("cpu")
    model._dtype = torch.float32
    prediction = torch.tensor([[[0.05, 0.15, 0.80]]])
    observed = torch.tensor([[0]])

    short_step = model._compute_posterior(
        prediction, observed, torch.tensor([[0.6]]), torch.tensor([[0.5]])
    )
    long_step = model._compute_posterior(
        prediction, observed, torch.tensor([[0.9]]), torch.tensor([[0.5]])
    )

    assert long_step[0, 0, 2] > short_step[0, 0, 2]
    assert long_step[0, 0, 0] < short_step[0, 0, 0]


def test_duo_model_output_is_normalized():
    model = _bare(DUO_BASE, temp=2.0)
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    result = model._process_model_output(logits, None, None)
    assert torch.allclose(result.exp().sum(-1), torch.ones(1, 1))


def test_duo_nll_per_token_has_expected_shape_and_finite_values():
    model = _bare(DUO_BASE, vocab_size=3)
    log_probs = torch.log_softmax(torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]]), -1)
    x0 = torch.tensor([[0, 2]])
    xt = torch.tensor([[0, 1]])
    result = model.nll_per_token(
        log_probs, xt, x0, torch.tensor([[0.6]]), torch.tensor([[-0.8]])
    )
    assert result.shape == x0.shape
    assert torch.isfinite(result).all()


def test_duo_ancestral_update_preserves_prompt(monkeypatch):
    model = _bare(
        DUO_BASE,
        vocab_size=3,
        p_nucleus=1.0,
        noise=lambda t: (None, torch.full_like(t, 0.5)),
    )
    model.forward = lambda x, _sigma: torch.log(torch.full((*x.shape, 3), 1 / 3))
    model._compute_posterior = lambda **kwargs: torch.nn.functional.one_hot(
        torch.full_like(kwargs["xt"], 2), 3
    ).float()
    tokens = torch.tensor([[1, 0, 1]])
    prompt_mask = torch.tensor([[1, 0, 1]])
    _, updated = model._ancestral_update(
        tokens, torch.tensor([[0.8]]), 0.2, prompt_mask=prompt_mask
    )
    assert torch.equal(updated, torch.tensor([[1, 2, 1]]))


def test_integral_clamps_boundaries_and_propagates_cached_gradient():
    gamma = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    data = {
        "gamma_min": 0.0,
        "gamma_max": 1.0,
        "num_points": 3,
        "pt": torch.tensor([0.1, 0.5, 0.9]),
        "grad_pt": torch.tensor([1.0, 2.0, 3.0]),
    }
    result = Integral.apply(gamma, data)
    assert torch.allclose(result, torch.tensor([0.1, 0.5, 0.9]))
    result.sum().backward()
    assert torch.equal(gamma.grad, torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    ("step", "expected"), [(0, 0.1), (15, 1.0), (30, 1e10)]
)
def test_duo_gumbel_temperature_curriculum(step, expected, monkeypatch):
    model = _bare(
        DUO,
        gumbel_tau_log10_start=1.0,
        gumbel_tau_log10_end=-1.0,
        curriculum_start=10,
        curriculum_end=20,
    )
    model._test_step = step
    monkeypatch.setattr(
        DUO, "global_step", property(lambda self: self._test_step), raising=False
    )
    assert model._compute_gumbel_tau_inverse() == pytest.approx(expected)
