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


def test_output_is_normalized():
    model = _bare(ContDiffOnehot, temp=2.0)
    result = model._process_model_output(torch.tensor([[[1.0, 2.0, 3.0]]]))
    assert torch.allclose(result.exp().sum(-1), torch.ones(1, 1))


def test_nll_is_token_cross_entropy():
    model = _bare(ContDiffOnehot)
    log_probs = torch.log(torch.tensor([[[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]]))
    result = model.nll_per_token(
        log_probs,
        x0_tokens=torch.tensor([[1, 0]]),
        alpha_t=torch.ones(1, 1),
        dalpha_t=torch.ones(1),
    )
    assert torch.allclose(result, -torch.log(torch.tensor([[0.7, 0.6]])))



def test_onehot_diffusion_q_xt_shape_and_finite_values():
    model = _bare(
        ContDiffOnehot,
        data_vocab_size=4,
        min_percentile=0.1,
        max_percentile=0.9,
        continuous_noise=torch.tensor([0.0, 1.0]),
        discrete_noise=torch.tensor([0.0, 1.0]),
    )
    tokens = torch.tensor([[0, 2], [1, 3]])
    result = model.q_xt(tokens, torch.tensor([[0.8], [0.4]]))
    assert result["xt"].shape == (2, 2, 4)
    assert torch.isfinite(result["xt"]).all()


def test_onehot_forward_process_has_requested_mean_and_standard_deviation():
    torch.manual_seed(6)
    sigma = 0.7
    model = _bare(ContDiffOnehot, data_vocab_size=4)
    model.get_continuous_from_discrete_noise = lambda time: torch.full_like(time, sigma)
    tokens = torch.zeros(40_000, 1, dtype=torch.long)

    noisy = model.q_xt(tokens, torch.full((tokens.shape[0], 1), 0.5))["xt"]
    residual = noisy - torch.nn.functional.one_hot(tokens, 4).float()

    assert residual.mean().item() == pytest.approx(0.0, abs=0.01)
    assert residual.std().item() == pytest.approx(sigma, abs=0.01)
