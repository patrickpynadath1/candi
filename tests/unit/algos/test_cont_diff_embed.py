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
    model = _bare(ContDiffEmbed, temp=2.0)
    result = model._process_model_output(torch.tensor([[[1.0, 2.0, 3.0]]]))
    assert torch.allclose(result.exp().sum(-1), torch.ones(1, 1))


def test_nll_is_token_cross_entropy():
    model = _bare(ContDiffEmbed)
    log_probs = torch.log(torch.tensor([[[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]]))
    result = model.nll_per_token(
        log_probs,
        x0_tokens=torch.tensor([[1, 0]]),
        alpha_t=torch.ones(1, 1),
        dalpha_t=torch.ones(1),
    )
    assert torch.allclose(result, -torch.log(torch.tensor([[0.7, 0.6]])))



def test_embedding_diffusion_noise_schedule_endpoints():
    model = _bare(ContDiffEmbed, sigma_min=0.1, sigma_max=10.0)
    result = model.get_continuous_from_discrete_noise(torch.tensor([0.0, 1.0]))
    assert torch.allclose(result, torch.tensor([0.1, 10.0]))


def test_embedding_diffusion_q_xt_clean_endpoint():
    class Backbone(torch.nn.Module):
        def get_embedding(self, tokens):
            return torch.nn.functional.one_hot(tokens, 4).float()

    model = _bare(
        ContDiffEmbed, backbone=Backbone(), sigma_min=0.0, sigma_max=1.0
    )
    model.get_continuous_from_discrete_noise = lambda t: torch.zeros_like(t)
    tokens = torch.tensor([[0, 2], [1, 3]])
    result = model.q_xt(tokens, torch.ones(2, 1))
    assert torch.equal(result["xt"], model.backbone.get_embedding(tokens))
    assert result["xt"].shape == (2, 2, 4)
