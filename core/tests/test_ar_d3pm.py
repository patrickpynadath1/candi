"""Causal cache equivalence and absorbing D3PM objective/lifecycle checks."""

from contextlib import nullcontext
from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from core.models.causal import CausalDiT
from core.objectives.d3pm import D3PMAbsorbingObjective
from core.samplers.autoregressive import AutoregressiveSampler
from core.training.module import TrainingModule


def config(method, *extra):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[1] / "new_configs")
    ):
        args = [
            f"+experiment={method}",
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=2",
            "model.dropout=0",
            "algorithm.length=10",
            "data.cache_dir=/tmp/core_ar_d3pm",
            "output_dir=/tmp/core_ar_d3pm",
            "training.warmup_steps=0",
        ]
        if method == "d3pm_absorb":
            args += ["model.cond_dim=16", "time_sampler.steps=10", "sampler.steps=4"]
        return compose(config_name="config", overrides=args + list(extra))


def model(device="cpu"):
    net = (
        CausalDiT(13, hidden_size=32, n_heads=4, n_blocks=2, dropout=0)
        .to(device)
        .eval()
    )
    with torch.no_grad():
        # A zero output head would hide attention/cache mistakes.
        net.output_layer.linear.weight.normal_(std=0.2)
    return net


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mixed", [False, True])
def test_cached_chunks_match_full_logits_and_are_causal(device, mixed):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    net = model(device)
    tokens = torch.randint(0, 13, (2, 9), device=device)
    amp = torch.autocast(device, dtype=torch.bfloat16) if mixed else nullcontext()
    with torch.no_grad(), amp:
        expected = net(tokens)
        cache = net.allocate_kv_cache(2, 9)
        parts = [
            net(tokens[:, :4], cache),
            net(tokens[:, 4:5], cache),
            net(tokens[:, 5:8], cache),
            net(tokens[:, 8:], cache),
        ]
        tolerance = 0.04 if mixed else 2e-6
        torch.testing.assert_close(
            torch.cat(parts, 1), expected, atol=tolerance, rtol=tolerance
        )
        assert cache.length == 9
        with pytest.raises(ValueError, match="capacity"):
            net(tokens[:, :1], cache)
        altered = tokens.clone()
        altered[:, 5:] = (altered[:, 5:] + 1) % 13
        torch.testing.assert_close(net(altered)[:, :5], expected[:, :5], atol=0, rtol=0)
        other = net.allocate_kv_cache(2, 9)
        assert other.layers[0].keys.data_ptr() != cache.layers[0].keys.data_ptr()
        torch.testing.assert_close(
            net(tokens[:, :3], other), expected[:, :3], atol=tolerance, rtol=tolerance
        )


def test_ar_cache_prefill_single_token_steps_and_ragged_prefixes():
    algo = instantiate(
        config("autoregressive", "parameterization.commitment=greedy").algorithm
    )
    with torch.no_grad():
        algo.model.output_layer.linear.weight.normal_(std=0.2)
    tokens = torch.randint(0, 12, (3, 10))
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :5] = True
    mask[2, :3] = True
    lengths = []
    hook = algo.model.blocks[0].attn_qkv.register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1])
    )
    cached, metadata = algo.generate(AutoregressiveSampler(), tokens, mask)
    hook.remove()
    uncached, reference = algo.generate(
        AutoregressiveSampler(use_cache=False), tokens, mask
    )
    assert torch.equal(cached, uncached)
    assert torch.equal(cached[mask], tokens[mask])
    assert lengths == [3] + [1] * 6 + [5] + [1] * 4
    assert metadata["tokens_processed"] < reference["tokens_processed"]
    assert metadata["nfe_per_sample"] == [7, 5, 7]
    with pytest.raises(ValueError, match="contiguous"):
        algo.generate(AutoregressiveSampler(), tokens, ~mask)


def test_ar_unconditional_bos_eos_and_shifted_training():
    algo = instantiate(config("ar").algorithm)
    tokens = torch.randint(0, 12, (2, 10))
    mask = torch.ones_like(tokens)
    loss = algo.loss(tokens, mask)["loss"]
    assert algo.loss(tokens, mask)["num_tokens"] == 18
    logits = algo.model(tokens[:, :-1])
    expected = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tokens[:, 1:])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    sampler = AutoregressiveSampler(bos_token_id=1, eos_token_id=2, pad_token_id=3)
    algo.predictor.commitment = "greedy"  # Sampling is forced by top-k independently.
    algo.predictor.top_k = 1
    with torch.no_grad():
        algo.model.output_layer.linear.bias[2] = 10
    result, stats = algo.generate(sampler, tokens, torch.zeros_like(tokens))
    assert result.tolist() == [[1, 2] + [3] * 8] * 2 and stats["nfe_per_sample"] == [
        1,
        1,
    ]
    with pytest.raises(ValueError, match="bos_token"):
        algo.generate(AutoregressiveSampler(), tokens, torch.zeros_like(tokens))


def test_d3pm_matches_reference_interior_and_finite_first_step():
    raw = torch.randn(2, 4, 5, requires_grad=True)
    prediction = raw.log_softmax(-1)
    reconstruction = torch.randn(2, 4, 5).log_softmax(-1)
    state = torch.tensor([[4, 0, 4, 1], [4, 4, 2, 4]])
    targets = torch.randint(0, 4, (2, 4))
    keep = torch.tensor([0.7, 0.3])
    T = 10
    dt = 1 / T
    time = (1 - keep[:, None]).clamp(0, 1 - 1e-4)
    a = keep[:, None] + torch.zeros_like(state)
    previous = time - dt + torch.zeros_like(state)
    p = prediction[..., 4].exp()
    logp = prediction.gather(-1, targets[..., None]).squeeze(-1)
    norm = torch.log(a * p / time + 1)
    expected = -reconstruction.gather(-1, targets[..., None]).squeeze(-1) + T * (
        (dt / time) * (norm - logp)
        + (1 - dt / time) * (norm - torch.log(previous * p / (time - dt) + 1))
    ) * state.eq(4)
    objective = D3PMAbsorbingObjective(4, T)
    actual = objective(prediction, state, targets, keep, reconstruction)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), raw, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), raw)[0],
    )
    first = objective(prediction, state, targets, torch.full((2,), 0.9), reconstruction)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("method", ["autoregressive", "d3pm_absorb"])
def test_configs_optimizer_checkpoint_and_generation(method):
    cfg = config(method)
    module = TrainingModule(cfg)
    algo = module.algorithm
    tokens = torch.randint(0, 12, (2, 10))
    mask = torch.ones_like(tokens)
    mask[:, :3] = 0
    optimizer = module.configure_optimizers()["optimizer"]
    loss = algo.loss(tokens, mask, training=True)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    optimizer.step()
    restored = TrainingModule(cfg)
    restored.load_state_dict(module.state_dict())
    algo.eval()
    restored.algorithm.eval()
    torch.manual_seed(3)
    expected = algo.loss(tokens, mask)["loss"]
    torch.manual_seed(3)
    actual = restored.algorithm.loss(tokens, mask)["loss"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    result, stats = algo.generate(instantiate(cfg.sampler), tokens, ~mask.bool())
    assert torch.equal(result[:, :3], tokens[:, :3]) and result.shape == tokens.shape
    if method == "d3pm_absorb":
        assert not result.eq(algo.process.mask_index).any()
        times = algo.time_sampler.sample(20, "cpu")
        torch.testing.assert_close(times * 10, (times * 10).round())
        assert times.min() > 0 and times.max() <= 1
        # Unconstrained prediction must still learn reconstruction at clean inputs.
        keep, _ = algo.schedule(torch.zeros(2))
        assert (
            algo.training_prediction(tokens, keep).exp().gather(-1, tokens[..., None])
            < 1
        ).all()
    else:
        assert stats["cache_used"]


def test_ar_text8_token_space_excludes_native_mask_without_synthetic_class():
    cfg = config("autoregressive", "task=text8")
    algo = instantiate(cfg.algorithm)
    assert algo.model.vocab_size == 35
    output = algo.predictor.training_posterior(torch.zeros(1, 2, 35), None)
    assert output[..., 4].isneginf().all()
