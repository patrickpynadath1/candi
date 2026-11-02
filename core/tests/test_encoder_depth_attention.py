"""Actual encoder K/V routing: semantics, gradients, visibility and cache parity."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from core.models.layers import regular_attention_multi_headed
from core.modifiers.block_diffusion.layout import make_layout
from core.modifiers.encoder_decoder.attention import EncoderDepthAttention
from core.sweeps.configuration import compose
from core.training.module import TrainingModule


def reference(q, k, v, ck, cv, mask):
    scores = torch.einsum("bqhd,bclhd->bhqcl", q, ck) / math.sqrt(q.shape[-1])
    depth = scores.softmax(-1)
    mixed_k = torch.einsum("bhqcl,bclhd->bqchd", depth, ck)
    mixed_v = torch.einsum("bhqcl,bclhd->bqchd", depth, cv)
    context_scores = torch.einsum("bqhd,bqchd->bhqc", q, mixed_k) / math.sqrt(
        q.shape[-1]
    )
    noisy_scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    weights = (
        torch.cat((context_scores, noisy_scores), -1)
        .masked_fill(~mask, -torch.inf)
        .softmax(-1)
    )
    out = torch.einsum("bhqc,bqchd->bqhd", weights[..., : ck.shape[1]], mixed_v)
    out = out + torch.einsum("bhqk,bkhd->bqhd", weights[..., ck.shape[1] :], v)
    return out.flatten(-2)


@pytest.mark.parametrize("chunk", [1, 3, 32])
def test_matches_explicit_query_dependent_mixtures_and_gradients(chunk):
    torch.manual_seed(19)
    q, k, v = [torch.randn(2, 4, 2, 4, requires_grad=True) for _ in range(3)]
    ck, cv = [torch.randn(2, 3, 3, 2, 4, requires_grad=True) for _ in range(2)]
    mask = torch.ones(2, 1, 4, 7, dtype=torch.bool)
    mask[:, :, :, 1] = False
    actual = EncoderDepthAttention(chunk)(q, k, v, (ck, cv), mask)
    expected = reference(q, k, v, ck, cv, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    inputs = (q, k, v, ck, cv)
    ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), inputs)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=3e-6, rtol=1e-4)


def test_single_depth_reduces_to_standard_attention_and_all_masked_is_zero():
    q, k, v = [torch.randn(1, 3, 2, 4) for _ in range(3)]
    ck, cv = [torch.randn(1, 2, 1, 2, 4) for _ in range(2)]
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    module = EncoderDepthAttention(2)
    actual = module(q, k, v, (ck, cv), mask)
    expected = regular_attention_multi_headed(
        q,
        torch.cat((ck[:, :, 0], k), 1),
        torch.cat((cv[:, :, 0], v), 1),
        attention_mask=mask,
    )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(module(q, k, v, (ck, cv), ~mask), torch.zeros_like(actual))


def config(recipe="encoder_kv_depth"):
    cfg = compose(
        recipe,
        [
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=2",
            "model.dropout=0",
            "task.length=8",
            "block_diffusion.enabled=true",
            "block_diffusion.block_size=2",
        ],
    )
    cfg.encoder_decoder = OmegaConf.create(
        {
            "_target_": "core.modifiers.encoder_decoder.EncoderDecoderModel",
            "encoder_layers": 2,
            "context_attention": {
                "_target_": "core.modifiers.encoder_decoder.attention.EncoderDepthAttention",
                "query_chunk_size": 2,
            },
        }
    )
    return cfg


@pytest.mark.parametrize(
    "recipe", ["encoder_kv_depth", "blockgen_uniform", "gaussian_diffusion_sudoku"]
)
def test_corruption_agnostic_loss_and_encoder_kv_gradients(recipe):
    algo = TrainingModule(config(recipe)).algorithm.train()
    # Bypass deliberately zero-initialized heads/adaLN gates for this gradient test.
    for module in algo.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.1)
    x = torch.randint(0, 9, (2, 8))
    loss = algo.loss(x, torch.ones_like(x, dtype=torch.bool), training=True)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    for block in algo.model.model.encoder:
        if hasattr(block, "attn_kv"):
            k, v = block.attn_kv.weight.grad.chunk(2)
        else:
            _, k, v = block.attn_qkv.weight.grad.chunk(3)
        assert k.norm() > 0 and v.norm() > 0


@pytest.mark.parametrize("checkpointed", [False, True])
def test_no_leakage_and_actual_encoder_keys_used(checkpointed):
    m = TrainingModule(config()).algorithm.model
    m.model.activation_checkpointing = checkpointed
    torch.nn.init.normal_(m.output_layer.linear.weight, std=0.1)
    # Open AdaLN-Zero gates to exercise context and cache semantics.
    for block in m.blocks:
        if block.adaLN:
            torch.nn.init.normal_(block.adaLN_modulation.weight, std=0.02)
    clean = torch.randint(0, 9, (2, 8))
    noisy = torch.randint(0, 9, (2, 8))
    time = torch.zeros(2)
    seen = []
    hook = m.model.encoder[0].register_forward_hook(
        lambda _m, _a, out: seen.append(out[1])
    )
    with m.clean_context(clean):
        baseline = m(noisy, time)
    hook.remove()
    # Keys and values returned by the actual encoder attention, not final states.
    with torch.no_grad():
        layout = make_layout(
            8, 2, clean.device, batch_size=2, clean_length=8, x0_causal=False
        )
        hidden = m.vocab_embed(clean)
        rotary = m.rotary_emb(hidden, position_ids=layout.position_ids[:, :8])
        context = m.model._encode(hidden, rotary, layout.attention_mask[:, :, :8, :8])
    torch.testing.assert_close(context[0][:, :, 0], seen[0][0])
    torch.testing.assert_close(context[1][:, :, 0], seen[0][1])
    changed = clean.clone()
    changed[:, 2:] = (changed[:, 2:] + 1) % 9
    with m.clean_context(changed):
        other = m(noisy, time)
    torch.testing.assert_close(baseline[:, :4], other[:, :4])
    changed = clean.clone()
    changed[:, :2] = (changed[:, :2] + 1) % 9
    with m.clean_context(changed):
        other = m(noisy, time)
    assert not torch.allclose(baseline[:, 2:4], other[:, 2:4])
    baseline.square().mean().backward()
    assert m.model.encoder[0].attn_qkv.weight.grad is not None


@torch.no_grad()
def test_incremental_cache_matches_parallel_and_recompute():
    m = TrainingModule(config()).algorithm.model.eval()
    torch.nn.init.normal_(m.output_layer.linear.weight, std=0.1)
    # Open AdaLN-Zero gates to exercise context and cache semantics.
    for block in m.blocks:
        if block.adaLN:
            torch.nn.init.normal_(block.adaLN_modulation.weight, std=0.02)
    clean = torch.randint(0, 9, (2, 8))
    noisy = torch.randint(0, 9, (2, 8))
    time = torch.zeros(2)
    with m.clean_context(clean):
        parallel = m(noisy, time)
    cache = {}
    lengths = []
    hook = m.model.encoder[0].register_forward_pre_hook(
        lambda _m, args: lengths.append(args[0].shape[1])
    )
    for start in (0, 2, 4, 6):
        with m.clean_context(clean[:, :start], offset=start, cache=cache):
            actual = m(noisy[:, start : start + 2], time)
            repeated = m(noisy[:, start : start + 2], time)
        torch.testing.assert_close(
            actual, parallel[:, start : start + 2], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(actual, repeated)
    hook.remove()
    assert lengths == [2, 2, 2]
    assert cache["encoder_depth_kv"][0].shape == (2, 6, 2, 4, 8)
    assert all(pair is None for pair in cache["decoder_context_kv"])


@pytest.mark.parametrize("layers", [1, 2])
def test_terminal_readout_matches_full_block_without_unused_parameters(layers):
    from core.models.layers import DDiTBlock
    from core.modifiers.encoder_decoder.readout import EncoderKeyValueReadout

    cfg = config()
    cfg.encoder_decoder.encoder_layers = layers
    optimized = TrainingModule(cfg).algorithm.model.eval()
    torch.nn.init.normal_(optimized.output_layer.linear.weight, std=0.1)
    # Open AdaLN-Zero gates to exercise context and cache semantics.
    for block in optimized.blocks:
        if block.adaLN:
            torch.nn.init.normal_(block.adaLN_modulation.weight, std=0.02)
    reference_model = TrainingModule(cfg).algorithm.model.eval()
    reference_model.load_state_dict(optimized.state_dict())
    readout = optimized.model.encoder[-1]
    assert isinstance(readout, EncoderKeyValueReadout)
    assert not hasattr(readout, "mlp") and not hasattr(readout, "attn_out")
    full = DDiTBlock(32, 4, adaLN=False, dropout=0)
    with torch.no_grad():
        full.norm1.weight.copy_(readout.norm1.weight)
        full.attn_qkv.weight[32:].copy_(readout.attn_kv.weight)
    reference_model.model.encoder[-1] = full
    assert sum(p.numel() for p in optimized.parameters()) < sum(
        p.numel() for p in reference_model.parameters()
    )
    clean = torch.randint(0, 9, (2, 8))
    noisy = torch.randint(0, 9, (2, 8))
    outputs = []
    for model in (optimized, reference_model):
        with model.clean_context(clean):
            outputs.append(model(noisy, torch.zeros(2)))
        outputs[-1].square().sum().backward()
    torch.testing.assert_close(*outputs, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        readout.attn_kv.weight.grad,
        full.attn_qkv.weight.grad[32:],
        atol=1e-5,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        optimized.vocab_embed.embedding.grad,
        reference_model.vocab_embed.embedding.grad,
        atol=1e-5,
        rtol=1e-4,
    )
