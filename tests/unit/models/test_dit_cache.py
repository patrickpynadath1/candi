from types import SimpleNamespace

import torch

from models.dit import DIT


def _config():
    return SimpleNamespace(
        algo=SimpleNamespace(causal_attention=True, time_conditioning=False),
        model=SimpleNamespace(
            hidden_size=8,
            cond_dim=8,
            n_heads=2,
            n_blocks=2,
            dropout=0.0,
            scale_by_sigma=False,
        ),
    )


def test_dit_positioned_cache_matches_full_and_supports_decoupled_rope():
    torch.manual_seed(11)
    model = DIT(_config(), vocab_size=7).eval()
    torch.nn.init.normal_(model.output_layer.linear.weight)
    torch.nn.init.normal_(model.output_layer.linear.bias)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 1, 5, 3, 4]])
    key_positions = torch.tensor([[0, 3, 1, 5, 2], [0, 2, 5, 1, 4]])
    query_positions = torch.tensor([[4, 1, 5, 2, 3], [3, 5, 1, 4, 2]])
    full, _ = model.forward_positioned(
        tokens,
        key_position_ids=key_positions,
        query_position_ids=query_positions,
    )
    cache = model.allocate_kv_cache(2, 5, device=torch.device("cpu"))
    pieces = []
    for start, end in ((0, 2), (2, 4), (4, 5)):
        output, returned = model.forward_positioned(
            tokens[:, start:end],
            key_position_ids=key_positions[:, start:end],
            query_position_ids=query_positions[:, start:end],
            cache=cache,
        )
        assert returned is cache
        pieces.append(output)
    assert cache.length == 5
    assert torch.allclose(full, torch.cat(pieces, 1), atol=2e-5, rtol=2e-5)


def test_standard_dit_forward_matches_explicit_sequential_positions():
    torch.manual_seed(13)
    model = DIT(_config(), vocab_size=7).eval()
    torch.nn.init.normal_(model.output_layer.linear.weight)
    tokens = torch.tensor([[1, 2, 3, 4]])
    positions = torch.arange(4).unsqueeze(0)
    standard = model(tokens, None)
    explicit, _ = model.forward_positioned(
        tokens, key_position_ids=positions, query_position_ids=positions
    )
    assert torch.allclose(standard, explicit, atol=2e-5, rtol=2e-5)
