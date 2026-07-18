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



def test_d3pm_output_masks_absorbing_token_when_subs_enabled():
    model = _bare(D3PMAbsorb, subs_masking=True, mask_index=2, neg_infinity=-1e6)
    result = model._process_model_output(torch.zeros(1, 2, 3), None, None)
    assert result.shape == (1, 2, 3)
    assert result[..., 2].max() < -1000
    assert torch.allclose(result.exp().sum(-1), torch.ones(1, 2))

