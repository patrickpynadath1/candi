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


def _config(*, causal=False, time_conditioning=True, reserve_mask_embedding=True):
    return SimpleNamespace(
        algo=SimpleNamespace(
            causal_attention=causal,
            time_conditioning=time_conditioning,
            reserve_mask_embedding=reserve_mask_embedding,
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




def test_cont_dit_token_and_embedding_paths():
    model = ContDIT(_config(), vocab_size=6)
    tokens = torch.tensor([[0, 1, 2], [2, 3, 4]])
    noise = torch.tensor([0.2, 0.7])
    token_logits = model(
        tokens,
        discrete_noise=noise,
        reveal_mask=torch.zeros_like(tokens),
        precond=False,
    )
    embeddings = model.get_embedding(tokens)
    embedding_logits = model(
        embeddings,
        discrete_noise=noise,
        reveal_mask=torch.zeros_like(tokens),
        is_embed=True,
        precond=False,
    )
    assert token_logits.shape == (2, 3, 5)
    assert embedding_logits.shape == token_logits.shape
    assert torch.isfinite(token_logits).all()
    assert torch.allclose(token_logits, embedding_logits)


def test_cont_dit_exact_noisy_endpoint_is_finite():
    model = ContDIT(_config(), vocab_size=6)
    tokens = torch.tensor([[0, 1, 2]])
    logits = model(
        tokens,
        discrete_noise=torch.ones(1),
        reveal_mask=torch.zeros_like(tokens),
        precond=False,
    )
    assert torch.isfinite(logits).all()


def test_cont_dit_uses_blurring_dit_input_and_output_geometry():
    model = ContDIT(_config(), vocab_size=6)
    data_table = torch.nn.functional.normalize(
        model.vocab_embed.embedding[:-1], dim=-1
    )

    assert model.input_projection.in_features == model.input_dim
    assert model.input_projection.out_features == model.dim
    assert model.output_projection.in_features == model.dim
    assert model.output_projection.out_features == model.output_dim
    assert model.output_embedding.shape == (6, model.output_dim)
    assert model.output_bias.shape == (6,)
    assert torch.allclose(data_table.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert model.output_embedding is not model.vocab_embed.embedding


def test_cont_dit_blurring_style_head_and_transformer_receive_gradients():
    torch.manual_seed(1)
    model = ContDIT(_config(), vocab_size=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(
            tokens,
            discrete_noise=torch.tensor([0.2, 0.7]),
            reveal_mask=torch.zeros_like(tokens),
            precond=False,
        )
        torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        ).backward()
        optimizer.step()

    for parameter in (
        model.input_projection.weight,
        model.blocks[0].attn_qkv.weight,
        model.output_projection.weight,
        model.output_embedding,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_cont_dit_mask_embedding_is_last_vocabulary_row():
    model = ContDIT(_config(), vocab_size=6)
    assert torch.equal(model.get_mask_embed(), model.vocab_embed.embedding[-1])


def test_cont_dit_without_mask_uses_final_token_and_returns_all_logits():
    model = ContDIT(
        _config(reserve_mask_embedding=False),
        vocab_size=35,
    )
    tokens = torch.tensor([[0, 34]])
    noise = torch.tensor([0.2])

    embeddings = model.get_embedding(tokens)
    logits = model(
        embeddings,
        discrete_noise=noise,
        reveal_mask=None,
        is_embed=True,
        precond=False,
    )

    assert embeddings.shape == (1, 2, 8)
    assert logits.shape == (1, 2, 35)
    with pytest.raises(RuntimeError, match="no reserved mask embedding"):
        model.get_mask_embed()
