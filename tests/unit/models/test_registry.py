# ruff: noqa: F401
# ruff: noqa: F401
from types import SimpleNamespace

import pytest
import torch
from catalogue import RegistryError

from models.dit import DIT
from models.dit_cfm import CFMDit
from models.dit_cont import ContDIT
from models.registry import models


def _config(*, causal=False, time_conditioning=True):
    return SimpleNamespace(
        algo=SimpleNamespace(
            causal_attention=causal, time_conditioning=time_conditioning
        ),
        model=SimpleNamespace(
            hidden_size=8,
            cond_dim=8,
            n_heads=2,
            n_blocks=0,
            dropout=0.0,
            scale_by_sigma=False,
        ),
    )




@pytest.mark.parametrize(
    ("name", "expected"),
    [("dit", DIT), ("cont_dit", ContDIT), ("cfm_dit", CFMDit)],
)
def test_active_backbones_are_registered(name, expected):
    assert models.get(name) is expected


def test_unknown_backbone_lookup_raises_descriptive_error():
    with pytest.raises(RegistryError, match="does_not_exist"):
        models.get("does_not_exist")

