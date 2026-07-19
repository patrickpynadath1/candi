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



def test_sedd_output_sets_observed_token_score_to_one():
    model = _bare(SEDDAbsorb)
    tokens = torch.tensor([[0, 2]])
    result = model._process_model_output(
        torch.ones(1, 2, 3), tokens, torch.tensor([0.7])
    )
    observed = result.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(observed, torch.zeros_like(observed))
    assert torch.isfinite(result).all()


def test_sedd_output_applies_exact_noise_and_vocabulary_scaling():
    model = _bare(SEDDAbsorb)
    tokens = torch.tensor([[0]])
    raw = torch.tensor([[[3.0, 4.0, 5.0]]])
    sigma = torch.tensor([torch.log(torch.tensor(2.0))])
    result = model._process_model_output(raw, tokens, sigma)

    # exp(sigma)-1 = 1 and there are vocab_size-1 alternative states.
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] == pytest.approx(4.0 - torch.log(torch.tensor(2.0)).item())
    assert result[0, 0, 2] == pytest.approx(5.0 - torch.log(torch.tensor(2.0)).item())
