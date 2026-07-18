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




def test_dit_forward_embedding_and_backward_on_cpu():
    model = DIT(_config(), vocab_size=6)
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    sigma = torch.tensor([0.2, 0.8])
    embeddings = model.get_embedding(tokens)
    logits = model(tokens, sigma)
    embedded_logits = model.forward_with_embeddings(embeddings, sigma)
    assert embeddings.shape == (2, 3, 8)
    assert logits.shape == (2, 3, 6)
    assert embedded_logits.shape == logits.shape
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert model.vocab_embed.embedding.grad is not None
    assert torch.isfinite(model.vocab_embed.embedding.grad).all()


def test_causal_dit_does_not_require_time_conditioning():
    model = DIT(_config(causal=True, time_conditioning=False), vocab_size=5)
    logits = model(torch.tensor([[0, 1, 2]]), None)
    assert logits.shape == (1, 3, 5)
    assert torch.isfinite(logits).all()

