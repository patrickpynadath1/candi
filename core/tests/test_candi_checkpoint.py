"""Regression tests for positional EMA import and strict parameter coverage."""

from collections import OrderedDict

import pytest
import torch
from torch import nn

from core.training.candi_checkpoint import legacy_candi_state, map_legacy_candi_state


def checkpoint():
    table = torch.zeros(3, 4)
    state = OrderedDict([
        ("backbone.vocab_embed.embedding", table),
        ("backbone.sigma_map.weight", torch.zeros(2, 4)),
        ("backbone.rotary_emb.inv_freq", torch.ones(2)),
        ("backbone.precond.embed_table.embedding", table),
        ("backbone.output_layer.weight", torch.zeros(3, 4)),
    ])
    return {"state_dict": state, "ema": {"shadow_params": [
        torch.full((3, 4), 1.), torch.full((2, 4), 2.), torch.full((3, 4), 3.),
    ]}}


def test_ema_skips_buffer_and_shared_parameter_alias():
    state = legacy_candi_state(checkpoint(), use_ema=True)
    assert state["backbone.vocab_embed.embedding"].eq(1).all()
    assert state["backbone.sigma_map.weight"].eq(2).all()
    assert state["backbone.output_layer.weight"].eq(3).all()
    assert state["backbone.precond.embed_table.embedding"] is state["backbone.vocab_embed.embedding"]
    assert state["backbone.rotary_emb.inv_freq"].eq(1).all()


def test_ema_missing_or_wrong_shape_fails_explicitly():
    saved = checkpoint()
    del saved["ema"]
    with pytest.raises(ValueError, match="no EMA"):
        legacy_candi_state(saved, use_ema=True)
    saved = checkpoint()
    saved["ema"]["shadow_params"][1] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        legacy_candi_state(saved, use_ema=True)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.empty(3, 4))

    def legacy_candi_state_map(self):
        return {"table": "backbone.vocab_embed.embedding"}


def test_mapper_rejects_unmapped_parameters_and_shape_mismatch():
    algo = nn.Module()
    algo.model = TinyModel()
    state = legacy_candi_state(checkpoint())
    mapped = map_legacy_candi_state(state, algo)
    assert torch.equal(mapped["model.table"], state["backbone.vocab_embed.embedding"])
    algo.model.extra = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="unmapped trainable"):
        map_legacy_candi_state(state, algo)
    del algo.model.extra
    state["backbone.vocab_embed.embedding"] = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        map_legacy_candi_state(state, algo)
