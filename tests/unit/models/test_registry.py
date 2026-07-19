# ruff: noqa: F401
# ruff: noqa: F401
from types import SimpleNamespace

import pytest
import torch

from models.dit import DIT
from models.dit_cfm import CFMDit
from models.dit_cont import ContDIT
from models.flm_dit import FLMDIT
from models.sphere_arch import SphereArch
from models.sphere_dit import SphereDiT
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
    [("dit", DIT), ("cont_dit", ContDIT), ("cfm_dit", CFMDit),
     ("flm_dit", FLMDIT), ("sphere_dit", SphereDiT),
     ("sphere_arch", SphereArch)],
)
def test_active_backbones_are_registered(name, expected):
    assert models.get(name) is expected
