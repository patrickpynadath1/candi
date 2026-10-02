"""Permutation targets, decoupled-position cache parity, and task composition."""

from contextlib import nullcontext
from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from core.generation_order import GenerationOrder
from core.models.causal import CausalDiT


def config(*overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=permuted_autoregressive",
                "task=text8",
                "task.length=12",
                "model.hidden_size=32",
                "model.n_heads=4",
                "model.n_blocks=2",
                "model.dropout=0",
                "output_dir=/tmp/permuted_test",
                *overrides,
            ],
        )


@pytest.mark.parametrize(
    "mode,block", [("random", None), ("sequential", None), ("block", 3)]
)
def test_order_policies(mode, block):
    order = GenerationOrder(mode, block).sample(4, 11, device="cpu")
    expected = torch.arange(11).expand_as(order)
    assert torch.equal(order.sort(-1).values, expected)
    if mode == "block":
        assert torch.equal(order // 3, expected // 3)
    assert torch.equal(
        GenerationOrder(mode, block, 1).sample(4, 11, device="cpu"), expected
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mixed", [False, True])
def test_positioned_cache_matches_full(device, mixed):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    net = (
        CausalDiT(35, hidden_size=32, n_heads=4, n_blocks=2, dropout=0)
        .to(device)
        .eval()
    )
    with torch.no_grad():
        net.output_layer.linear.weight.normal_(std=0.15)
    tokens = torch.randint(0, 35, (3, 11), device=device)
    queries = GenerationOrder().sample(3, 11, device=device) + 1
    keys = torch.cat((torch.zeros_like(queries[:, :1]), queries[:, :-1]), 1)
    with (
        torch.no_grad(),
        torch.autocast(device, dtype=torch.bfloat16) if mixed else nullcontext(),
    ):
        # nullcontext branch still needs no_grad for cache operations.
        with torch.no_grad():
            full = net(tokens, key_position_ids=keys, query_position_ids=queries)
            cache = net.allocate_kv_cache(3, 11)
            chunks = []
            for start, end in [(0, 4), (4, 5), (5, 9), (9, 11)]:
                chunks.append(
                    net(
                        tokens[:, start:end],
                        cache=cache,
                        key_position_ids=keys[:, start:end],
                        query_position_ids=queries[:, start:end],
                    )
                )
            torch.testing.assert_close(
                torch.cat(chunks, 1),
                full,
                atol=0.04 if mixed else 2e-6,
                rtol=0.04 if mixed else 2e-5,
            )
            changed = tokens.clone()
            changed[:, 5:] = (changed[:, 5:] + 1) % 35
            torch.testing.assert_close(
                net(changed, key_position_ids=keys, query_position_ids=queries)[:, :5],
                full[:, :5],
                atol=0,
                rtol=0,
            )


def test_training_matches_reference_and_backpropagates():
    algo = instantiate(config().algorithm)
    tokens = torch.randint(8, 35, (3, 12))
    tokens[:, 0] = 2
    valid = torch.ones_like(tokens)
    valid[0, 3] = 0
    torch.manual_seed(5)
    order = algo.order.sample(3, 11, device="cpu")
    targets = tokens[:, 1:].gather(1, order)
    inputs = torch.cat((tokens[:, :1], targets[:, :-1]), 1)
    keys = torch.cat((torch.zeros_like(order[:, :1]), order[:, :-1] + 1), 1)
    logits = algo.model(inputs, key_position_ids=keys, query_position_ids=order + 1)
    logits[..., 4] = -torch.inf
    nll = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    weight = valid[:, 1:].gather(1, order)
    torch.manual_seed(5)
    loss = algo.loss(tokens, valid)
    torch.testing.assert_close(loss["loss"], (nll * weight).sum() / weight.sum())
    loss["loss"].backward()
    assert algo.model.output_layer.linear.weight.grad.abs().sum() > 0


def test_sampling_cached_uncached_and_observed_conditioning():
    cfg = config(
        "parameterization.commitment=greedy", "algorithm.condition_on_observed=true"
    )
    algo = instantiate(cfg.algorithm).eval()
    with torch.no_grad():
        algo.model.output_layer.linear.weight.normal_(std=0.15)
    sampler = instantiate(cfg.sampler)
    sampler.configure_special_tokens(bos_token_id=2, eos_token_id=3, pad_token_id=5)
    tokens = torch.randint(8, 35, (3, 12))
    fixed = torch.zeros_like(tokens, dtype=torch.bool)
    fixed[0, 3] = True
    fixed[1, 4:8] = True
    torch.manual_seed(11)
    cached, metadata = algo.generate(sampler, tokens, fixed)
    sampler.use_cache = False
    torch.manual_seed(11)
    uncached, reference = algo.generate(sampler, tokens, fixed)
    assert torch.equal(cached, uncached)
    assert torch.equal(cached[fixed], tokens[fixed])
    assert metadata["tokens_processed"] == 33
    assert reference["tokens_processed"] == 3 * sum(range(1, 12))
    assert metadata["nfe_per_sample"] == [11] * 3
    assert (cached[:, 0] == 2).all()
    assert not (cached == 4).any()
    algo.condition_on_observed = False
    with pytest.raises(ValueError, match="condition_on_observed"):
        algo.generate(sampler, tokens, fixed)


def test_observed_partition_and_checkpoint():
    order = torch.tensor([[3, 0, 2, 1]])
    observed = torch.tensor([[True, False, True, False]])
    assert GenerationOrder.observed_first(order, observed).tolist() == [[0, 2, 3, 1]]
    cfg = config()
    first = instantiate(cfg.algorithm)
    restored = instantiate(cfg.algorithm)
    restored.load_state_dict(first.state_dict())
    tokens = torch.randint(8, 35, (2, 12))
    mask = torch.ones_like(tokens)
    torch.manual_seed(9)
    a = first.loss(tokens, mask)["loss"]
    torch.manual_seed(9)
    b = restored.loss(tokens, mask)["loss"]
    torch.testing.assert_close(a, b)


def test_positioned_forward_matches_reference_attention():
    # Explicit original decoupled-RoPE equations, using matrix attention rather
    # than the implementation's SDPA/cache path. Nonzero head detects mistakes.
    net = CausalDiT(35, hidden_size=32, n_heads=4, n_blocks=2, dropout=0).eval()
    with torch.no_grad():
        net.output_layer.linear.weight.normal_(std=0.15)
        tokens = torch.randint(8, 35, (2, 7))
        queries = torch.tensor([[4, 1, 6, 3, 2, 7, 5], [7, 5, 3, 1, 6, 2, 4]])
        keys = torch.cat((torch.zeros_like(queries[:, :1]), queries[:, :-1]), 1)
        hidden = net.vocab_embed(tokens)

        def rotate(x, positions):
            inverse = 10000 ** (-torch.arange(0, 8, 2).float() / 8)
            angle = torch.einsum("bs,d->bsd", positions.float(), inverse)
            angle = torch.cat((angle, angle), -1).unsqueeze(2)
            a, b = x.chunk(2, -1)
            return x * angle.cos() + torch.cat((-b, a), -1) * angle.sin()

        for block in net.blocks:
            q, k, v = (
                block.attn_qkv(block.norm1(hidden)).reshape(2, 7, 3, 4, 8).unbind(2)
            )
            q, k = rotate(q, queries).transpose(1, 2), rotate(k, keys).transpose(1, 2)
            scores = q @ k.transpose(-1, -2) / (8**0.5)
            scores = scores.masked_fill(
                torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf
            )
            attended = (
                (scores.softmax(-1) @ v.transpose(1, 2))
                .transpose(1, 2)
                .reshape(2, 7, 32)
            )
            hidden = hidden + block.attn_out(attended)
            hidden = hidden + block.mlp(block.norm2(hidden))
        expected = net.output_layer(hidden, None)
        actual = net(tokens, key_position_ids=keys, query_position_ids=queries)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("size", [0, -1, True, 2.5, None])
def test_invalid_block_width_rejected(size):
    with pytest.raises(ValueError, match="positive block_size"):
        GenerationOrder("block", size)


def test_eos_does_not_stop_spatial_filling():
    cfg = config("parameterization.commitment=greedy")
    algo = instantiate(cfg.algorithm)
    with torch.no_grad():
        algo.model.output_layer.linear.bias[3] = 10
    sampler = instantiate(cfg.sampler)
    sampler.configure_special_tokens(bos_token_id=2, eos_token_id=3, pad_token_id=5)
    tokens = torch.zeros(2, 12, dtype=torch.long)
    result, metadata = algo.generate(
        sampler, tokens, torch.zeros_like(tokens, dtype=torch.bool)
    )
    assert (result[:, 1:] == 3).all()
    assert metadata["nfe_per_sample"] == [11, 11]
