import pytest
import torch

from models.common import (
    DDiTBlock,
    DDiTBlockCausal,
    DDiTFinalLayer,
    EmbeddingLayer,
    EmbeddingLayerMixture_norm,
    Preconditioning,
    PositionRotary,
    LayerKVCache,
    Rotary,
    TimestepEmbedder,
)


def test_embedding_layer_mixture_path_matches_softmax_weighting():
    layer = EmbeddingLayer(dim=4, vocab_dim=5)
    tokens = torch.tensor([[0, 2, 4]])
    one_hot = torch.nn.functional.one_hot(tokens, 5).float()
    expected = torch.einsum("blv,ve->ble", torch.softmax(one_hot, -1), layer.embedding)
    assert torch.allclose(layer(one_hot), expected)
    assert layer(tokens).shape == expected.shape


def test_mixture_embedding_reserves_last_row_for_mask():
    layer = EmbeddingLayerMixture_norm(dim=4, vocab_dim=5)
    tokens = torch.tensor([[0, 3]])
    result = layer(tokens)
    assert result.shape == (1, 2, 4)
    assert torch.allclose(result.norm(dim=-1), torch.ones(1, 2))
    with pytest.raises(IndexError):
        layer(torch.tensor([[4]]))


def test_timestep_embedder_shape_and_finite_values():
    embedder = TimestepEmbedder(hidden_size=8)
    result = embedder(torch.tensor([0.0, 0.5, 1.0]))
    assert result.shape == (3, 8)
    assert torch.isfinite(result).all()


def test_rotary_embedding_matches_sequence_and_head_dimensions():
    rotary = Rotary(dim=4)
    cos, sin = rotary(torch.zeros(2, 5, 8))
    assert cos.shape == (1, 5, 3, 1, 4)
    assert sin.shape == (1, 5, 3, 1, 4)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_position_rotary_supports_batch_specific_positions():
    rotary = PositionRotary(dim=4)
    positions = torch.tensor([[0, 2, 1], [3, 1, 0]])
    cos, sin = rotary(positions)
    assert cos.shape == sin.shape == (2, 3, 1, 4)
    assert torch.allclose(cos[0, 0], cos[1, 2])
    assert not torch.allclose(cos[0, 1], cos[1, 0])


def test_layer_kv_cache_appends_and_checks_capacity():
    cache = LayerKVCache.allocate(
        2, 3, 2, 4, device=torch.device("cpu"), dtype=torch.float32
    )
    keys = torch.randn(2, 2, 2, 4)
    values = torch.randn_like(keys)
    all_keys, all_values = cache.append(keys, values)
    assert cache.length == 2
    assert torch.equal(all_keys, keys)
    assert torch.equal(all_values, values)
    with pytest.raises(ValueError, match="capacity exceeded"):
        cache.append(keys, values)


def test_positioned_causal_block_cache_matches_full_forward():
    torch.manual_seed(7)
    block = DDiTBlockCausal(dim=8, n_heads=2, dropout=0.0).eval()
    hidden = torch.randn(2, 5, 8)
    positions = torch.tensor([[0, 3, 1, 4, 2], [2, 0, 4, 1, 3]])
    rotary = PositionRotary(4)
    full = block.forward_positioned(hidden, rotary(positions), rotary(positions))
    cache = LayerKVCache.allocate(2, 5, 2, 4, device=hidden.device, dtype=hidden.dtype)
    chunks = []
    for start, end in ((0, 2), (2, 3), (3, 5)):
        chunks.append(
            block.forward_positioned(
                hidden[:, start:end],
                rotary(positions[:, start:end]),
                rotary(positions[:, start:end]),
                cache,
            )
        )
    assert torch.allclose(full, torch.cat(chunks, dim=1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("ada_ln", [False, True])
def test_noncausal_block_forward_backward(ada_ln):
    block = DDiTBlock(dim=8, n_heads=2, cond_dim=8, adaLN=ada_ln, dropout=0.0)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    rotary = Rotary(4)(hidden)
    condition = torch.randn(2, 8) if ada_ln else None
    output = block(hidden, rotary, c=condition)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_causal_block_forward_backward():
    block = DDiTBlockCausal(dim=8, n_heads=2, dropout=0.0)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = block(hidden, Rotary(4)(hidden), c=None)
    assert output.shape == hidden.shape
    output.sum().backward()
    assert torch.isfinite(hidden.grad).all()


def test_final_layer_shape_and_conditioning():
    layer = DDiTFinalLayer(hidden_size=8, out_channels=5, cond_dim=8, adaLN=True)
    result = layer(torch.randn(2, 3, 8), torch.randn(2, 8))
    assert result.shape == (2, 3, 5)
    assert torch.equal(result, torch.zeros_like(result))


def test_preconditioning_clean_positions_and_noisy_mixture():
    embeddings = EmbeddingLayerMixture_norm(dim=4, vocab_dim=5)
    precondition = Preconditioning(embeddings, mixed_coeff=0.25)
    x = torch.randn(1, 3, 4)
    reveal = torch.tensor([[1, 0, 1]])
    result = precondition(x, continuous_noise=torch.tensor([0.5]), reveal_mask=reveal)
    assert torch.equal(result[:, [0, 2]], x[:, [0, 2]])
    assert result.shape == x.shape
    assert torch.isfinite(result).all()
