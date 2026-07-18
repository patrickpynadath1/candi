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
            n_blocks=1,
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
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), torch.tensor([[1, 2, 3], [2, 3, 4]]).flatten()
    )
    loss.backward()
    assert model.output_layer.linear.weight.grad is not None
    assert torch.count_nonzero(model.output_layer.linear.weight.grad) > 0


def test_cfm_dit_accepts_full_vocabulary_continuous_states():
    model = CFMDit(_config(), vocab_size=5)
    state = torch.randn(1, 3, 5)
    logits = model(state, torch.tensor([0.4]))
    assert logits.shape == state.shape
    assert torch.isfinite(logits).all()
