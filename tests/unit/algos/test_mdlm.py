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



def test_mdlm_keeps_unmasked_tokens_deterministic():
    model = _bare(MDLM, temp=1.0, mask_index=3, neg_infinity=-1e6)
    xt = torch.tensor([[1, 3]])
    output = model._process_model_output(torch.zeros(1, 2, 4), xt, None)
    assert output.shape == (1, 2, 4)
    assert output[0, 0].argmax() == 1
    assert output[0, 0, 1] == 0


def test_mdlm_nll_per_token_matches_closed_form():
    model = _bare(MDLM)
    log_probs = torch.log(torch.tensor([[[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]]]))
    x0 = torch.tensor([[1, 0]])
    alpha = torch.tensor([[0.75]])
    dalpha = torch.tensor([-0.5])
    result = model.nll_per_token(log_probs, torch.zeros_like(x0), x0, alpha, dalpha)
    expected = torch.tensor([[torch.log(torch.tensor(0.6)), torch.log(torch.tensor(0.7))]])
    expected = expected * dalpha / (1 - alpha)
    assert torch.allclose(result, expected)


def test_mdlm_score_is_finite_nonnegative_and_preserves_shape():
    model = _bare(MDLM, mask_index=2, neg_infinity=-1e6)
    model.forward = lambda x, _sigma: torch.log_softmax(torch.zeros(*x.shape, 3), -1)
    tokens = torch.tensor([[2, 1]])
    score = model._get_score(tokens, torch.tensor([[0.7]]))
    assert score.shape == (1, 2, 3)
    assert torch.isfinite(score).all()
    assert torch.all(score >= 0)
    assert score[0, 1, 1] == pytest.approx(1.0)


def test_mdlm_score_matches_masked_and_observed_state_formula():
    model = _bare(MDLM, mask_index=2, neg_infinity=-1e6)
    probabilities = torch.tensor([[[0.2, 0.8, 0.0], [0.6, 0.4, 0.0]]])
    model.forward = lambda x, _sigma: probabilities.clamp_min(1e-30).log()
    tokens = torch.tensor([[2, 1]])
    sigma = torch.tensor([[torch.log(torch.tensor(3.0))]])

    score = model._get_score(tokens, sigma)

    # k = exp(-sigma) / (1-exp(-sigma)) = 1/2.
    assert torch.allclose(score[0, 0], torch.tensor([0.1, 0.4, 1.0]))
    assert score[0, 1, 1] == pytest.approx(1.0)
    assert score[0, 1, 2] == pytest.approx(2.0)
    assert score[0, 1, 0] == pytest.approx(0.0)
