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




def test_cfm_dit_forward_and_backward_on_cpu():
    model = CFMDit(_config(), vocab_size=5)
    tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])
    logits = model(tokens, torch.tensor([0.2, 0.8]))
    assert logits.shape == (2, 3, 5)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert model.vocab_embed.embedding.grad is not None

