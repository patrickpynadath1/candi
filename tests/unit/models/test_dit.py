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




def test_dit_token_and_embedding_paths_are_equivalent_with_a_real_block():
    torch.manual_seed(0)
    model = DIT(_config(), vocab_size=6)
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    sigma = torch.tensor([0.2, 0.8])
    embeddings = model.get_embedding(tokens)
    logits = model(tokens, sigma)
    embedded_logits = model.forward_with_embeddings(embeddings, sigma)
    assert embeddings.shape == (2, 3, 8)
    assert logits.shape == (2, 3, 6)
    assert torch.allclose(embedded_logits, logits)


def test_dit_training_reaches_transformer_parameters_after_zero_init_warmup():
    torch.manual_seed(1)
    model = DIT(_config(), vocab_size=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 0]])
    sigma = torch.tensor([0.2, 0.8])

    # The DiT output and adaptive gates intentionally start at zero. The first
    # update opens the output path, the second opens the adaptive gates, and the
    # third must reach the transformer weights behind those gates.
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(tokens, sigma).flatten(0, 1), targets.flatten()
        )
        loss.backward()
        optimizer.step()

    attention_grad = model.blocks[0].attn_qkv.weight.grad
    assert attention_grad is not None
    assert torch.isfinite(attention_grad).all()
    assert torch.count_nonzero(attention_grad) > 0


def test_causal_dit_does_not_require_time_conditioning():
    model = DIT(_config(causal=True, time_conditioning=False), vocab_size=5)
    logits = model(torch.tensor([[0, 1, 2]]), None)
    assert logits.shape == (1, 3, 5)
    assert torch.isfinite(logits).all()


def test_causal_dit_future_tokens_cannot_change_earlier_logits():
    torch.manual_seed(2)
    model = DIT(_config(causal=True, time_conditioning=False), vocab_size=6).eval()
    torch.nn.init.normal_(model.output_layer.linear.weight)
    original = torch.tensor([[0, 1, 2, 3]])
    changed_future = torch.tensor([[0, 1, 5, 4]])

    original_logits = model(original, None)
    changed_logits = model(changed_future, None)

    assert torch.allclose(original_logits[:, :2], changed_logits[:, :2], atol=1e-6)
    assert not torch.allclose(original_logits[:, 2:], changed_logits[:, 2:])
