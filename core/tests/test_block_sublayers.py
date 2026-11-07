from functools import partial

import pytest
import torch
from torch import nn
from hydra.utils import instantiate

from core.models.layers import DDiTBlock, Rotary
from core.models.sublayers import CenteredMLP, ExpandedAttention, PositionwiseMLP
from core.sweeps.configuration import compose


def test_default_mlp_preserves_weights_and_computation():
    old = nn.Sequential(
        nn.Linear(32, 128), nn.GELU(approximate="tanh"), nn.Linear(128, 32)
    )
    new = PositionwiseMLP(32)
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, 7, 32)
    torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)


def test_expanded_attention_mixes_positions_and_respects_mask():
    torch.manual_seed(12)
    module = ExpandedAttention(32, 4)
    x = torch.randn(2, 7, 32)
    rotary = Rotary(8)(x)
    causal = torch.ones(7, 7, dtype=torch.bool).tril()
    changed = x.clone()
    changed[:, -1] += torch.randn(32) * 10
    assert not torch.allclose(
        module(x, rotary_cos_sin=rotary)[:, 0],
        module(changed, rotary_cos_sin=rotary)[:, 0],
    )
    torch.testing.assert_close(
        module(x, rotary_cos_sin=rotary, attention_mask=causal)[:, :-1],
        module(changed, rotary_cos_sin=rotary, attention_mask=causal)[:, :-1],
    )
    assert sum(p.numel() for p in ExpandedAttention(512, 8).parameters()) == 2097152


@pytest.mark.parametrize("adaln", [False, True])
def test_second_attention_block_gradients_and_cache_guard(adaln):
    block = DDiTBlock(
        32,
        4,
        adaln,
        16,
        dropout=0,
        sublayer_factory=partial(ExpandedAttention, expansion=2),
    )
    if adaln:
        nn.init.normal_(block.adaLN_modulation.weight, std=0.1)
    x = torch.randn(2, 7, 32)
    rotary = Rotary(8)(x)
    y = block(x, rotary, torch.randn(2, 16))
    y.square().mean().backward()
    assert y.shape == x.shape
    assert block.mlp.qkv.weight.grad.abs().sum() > 0
    assert torch.isfinite(block.mlp.qkv.weight.grad).all()
    with pytest.raises(ValueError, match="KV caching"):
        block(x, rotary, return_key_value=True)


@pytest.mark.parametrize("choice", ["mlp", "expanded_attention", "centered_mlp"])
def test_masked_recipe_second_sublayer_training_and_generation(choice):
    cfg = compose(
        overrides=["+sweep=masked_attention_sudoku", f"block_sublayer={choice}"]
    )
    cfg.model.hidden_size = 32
    cfg.model.cond_dim = 16
    cfg.model.n_heads = 4
    cfg.model.n_blocks = 2
    algo = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 10, (2, 8))
    loss = algo.loss(
        tokens, torch.ones_like(tokens, dtype=torch.bool), step=0, training=True
    )["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, :2] = True
    result, _ = algo.eval().generate(instantiate(cfg.sampler, steps=2), tokens, prompt)
    assert result.shape == tokens.shape
    assert torch.equal(result[prompt], tokens[prompt])


def test_centering_zero_gate_equivalence_gradient_and_mean():
    torch.manual_seed(42)
    baseline = PositionwiseMLP(32)
    torch.manual_seed(42)
    centered = CenteredMLP(32)
    x = torch.randn(2, 7, 32)
    torch.testing.assert_close(centered(x), baseline(x), rtol=0, atol=0)
    centered(x).square().sum().backward()
    assert centered.centering_strength.grad.abs() > 0
    with torch.no_grad():
        centered.centering_strength.fill_(1)
    a = centered.up(x)
    expected = centered.down(centered.activation(a - a.mean(-2, keepdim=True)))
    torch.testing.assert_close(centered(x), expected)
    changed = x.clone()
    changed[:, -1] += torch.randn(32) * 10
    assert not torch.allclose(centered(x)[:, 0], centered(changed)[:, 0])
    with pytest.raises(ValueError, match="unmasked"):
        centered(x, attention_mask=torch.ones(7, 7, dtype=torch.bool).tril())
    block = DDiTBlock(32, 4, False, 16, sublayer_factory=CenteredMLP)
    with pytest.raises(ValueError, match="KV caching"):
        block(x, Rotary(8)(x), return_key_value=True)
