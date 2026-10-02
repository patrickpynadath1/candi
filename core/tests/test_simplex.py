"""Copied legacy simplex regression tests, adapted to isolated core interfaces."""

from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from core.noise_schedules.simplex import concentration, SimplexSchedule
from core.processes.simplex_math import (
    corruption_parameters,
    log_dirichlet_sample,
    reverse_bridge,
)
from core.processes.dirichlet import DirichletProcess
from core.samplers.simplex import SimplexBridgeSampler
from core.samplers.simplex_utils import guided_log_probs
from core.time_sampling.loss_derivative import LossDerivativeTimeSampler


def tiny_config(sc=False, **overrides):
    entries = [
        "+experiment=sudoku_simplex",
        "model.hidden_size=16",
        "model.n_heads=2",
        "model.n_blocks=1",
        "model.cond_dim=8",
        "model.dropout=0",
        "algorithm.length=6",
        f"self_conditioning.enabled={str(sc).lower()}",
    ]
    entries.extend(f"{key}={value}" for key, value in overrides.items())
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
        version_base=None,
    ):
        return compose(config_name="config", overrides=entries)


@pytest.mark.parametrize("churn", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("kind", ["constant", "constant_linear", "ddim"])
def test_oracle_reverse_bridge_marginals(churn, kind):
    n = 12000
    s, t = (
        torch.full((n,), 0.25, dtype=torch.float64),
        torch.full((n,), 0.85, dtype=torch.float64),
    )
    cs, ct = concentration(s, kind), concentration(t, kind)
    prior = torch.full((3,), 1 / 3, dtype=torch.float64)
    tokens = torch.zeros(n, 1, dtype=torch.long)
    clean = torch.nn.functional.one_hot(tokens, 3).double()
    bs, bt = (
        corruption_parameters(clean, s, cs, prior),
        corruption_parameters(clean, t, ct, prior),
    )
    pt = log_dirichlet_sample(bt)
    ps = reverse_bridge(pt, tokens, s, t, cs, ct, prior, churn).exp()[:, 0]
    mean = bs[0, 0] / cs[0]
    variance = mean * (1 - mean) / (cs[0] + 1)
    torch.testing.assert_close(ps.mean(0), mean, atol=0.012, rtol=0)
    torch.testing.assert_close(ps.var(0), variance, atol=0.008, rtol=0)
    torch.testing.assert_close(ps.sum(-1), torch.ones(n, dtype=ps.dtype))


def test_log_space_tiny_shapes_and_exact_faces():
    beta = torch.tensor([[1e-8, 2e-8, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    logp = log_dirichlet_sample(beta)
    assert torch.isfinite(logp[0, :2]).all()
    assert torch.isneginf(logp[:, 2]).all()
    torch.testing.assert_close(
        logp.exp()[1], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(2, dtype=torch.float64))


def test_adaptive_clock_concentrates_on_loss_derivative_and_reloads():
    sampler = LossDerivativeTimeSampler(
        adaptive=True, buffer_size=2000, bins=50, warmup=10, refit_every=5
    )
    t = torch.linspace(0, 1, 2000)
    loss = t.square()[:, None]
    sampler.observe(t, loss, None, 0)
    assert not sampler.ready and sampler.count == 2000
    sampler.observe(t, loss, None, 10)
    assert sampler.ready
    # q(t)=2t -> inverse CDF sqrt(u).
    torch.testing.assert_close(
        sampler(torch.tensor([0.25, 0.64])),
        torch.tensor([0.5, 0.8]),
        atol=0.025,
        rtol=0,
    )
    restored = LossDerivativeTimeSampler(adaptive=True, buffer_size=2000)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.cdf, sampler.cdf)
    with pytest.raises(ValueError, match="fitted"):
        LossDerivativeTimeSampler().sampling_grid(3, "adaptive", "cpu")


def test_reverse_bridge_supports_reference_on_a_face():
    prior = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    t = torch.tensor([1.0], dtype=torch.float64)
    s = torch.tensor([0.4], dtype=torch.float64)
    c = torch.ones_like(t)
    pt = log_dirichlet_sample(prior[None, None])
    result = reverse_bridge(pt, torch.tensor([[0]]), s, t, c, c, prior, 0.2)
    assert torch.isneginf(result[..., 2]).all()
    torch.testing.assert_close(
        result.logsumexp(-1), torch.zeros(1, 1, dtype=torch.float64)
    )


@pytest.mark.parametrize("churn", [0.0, 0.2, 1.0])
def test_full_sampler_recovers_known_data_distribution(churn):
    """Exact Bayes denoiser checks the entire chain, not just one bridge.

    With uniform reference, all class-conditional Dirichlet normalizers
    are equal. Bayes logits are log(data prior) + c(t)(1-t) log(P_t).
    Every reverse step must preserve this known mixture marginal.
    """
    torch.manual_seed(27)
    config = tiny_config(**{"algorithm.length": 1})
    model = instantiate(config.algorithm).eval()
    data_prior = torch.full(
        (model.process.vocab_size,), 0.1 / (model.process.vocab_size - 2)
    )
    data_prior[:2] = torch.tensor([0.7, 0.2])

    def exact_posterior(log_state, t):
        strength = model.schedule(t).concentration * (1 - t.double())
        return (data_prior.log() + strength[:, None, None] * log_state).log_softmax(-1)

    model.sampling_prediction = exact_posterior
    sampler = SimplexBridgeSampler(steps=12, churn=churn)
    samples, metadata = sampler.sample(
        model,
        torch.zeros(2048, 1, dtype=torch.long),
        torch.zeros(2048, 1, dtype=torch.bool),
    )
    frequencies = samples.flatten().bincount(minlength=model.process.vocab_size) / 2048
    torch.testing.assert_close(frequencies, data_prior, atol=0.035, rtol=0)
    assert metadata["nfe"] == 12


@pytest.mark.parametrize("sc", [False, True])
@pytest.mark.parametrize("input_mode", ["expectation", "argmax"])
def test_real_loss_gradient_and_prompt_preservation(sc, input_mode):
    config = tiny_config(sc, **{"model.input_mode": input_mode})
    model = instantiate(config.algorithm)
    x = torch.randint(1, 10, (2, 6))
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :3] = False
    loss = model.loss(x, mask, training=True)["loss"]
    loss.backward()
    grad = model.model.output_layer.linear.weight.grad
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm() > 0
    model.eval()
    for churn in (0.0, 0.2, 1.0):
        samples, metadata = model.generate(
            SimplexBridgeSampler(steps=3, churn=churn), x, ~mask
        )
        assert metadata["nfe"] == 3
        assert torch.equal(samples[:, :3], x[:, :3])
    restored = instantiate(config.algorithm)
    restored.load_state_dict(model.state_dict())


def test_guided_and_confidence_sampling_preserve_condition_and_count_calls():
    config = tiny_config(True, **{"algorithm.condition_dropout": 0.1})
    model = instantiate(config.algorithm).eval()
    x = torch.randint(1, 10, (2, 6))
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :3] = True
    calls = []
    hook = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs.get("self_condition")),
        with_kwargs=True,
    )
    sampler = SimplexBridgeSampler(steps=3, guidance_scale=2.0, confidence_power=1.0)
    samples, metadata = sampler.sample(model, x, mask)
    assert metadata["nfe"] == 6
    assert torch.equal(samples[:, :3], x[:, :3])
    assert calls[0] is None and calls[1] is None
    assert calls[2] is not None and calls[3] is not None
    sampler.sample(model, x, mask)
    assert calls[6] is None and calls[7] is None
    hook.remove()


def test_guidance_identities():
    a, b = torch.randn(2, 3).log_softmax(-1), torch.randn(2, 3).log_softmax(-1)
    torch.testing.assert_close(guided_log_probs(a, b, 0), a)
    torch.testing.assert_close(guided_log_probs(a, b, 1), b)


@pytest.mark.parametrize("kind", ["constant", "constant_linear", "ddim"])
def test_clean_endpoint_and_prompt_dropout(kind):
    process = DirichletProcess(3)
    schedule = SimplexSchedule(kind=kind)
    tokens = torch.tensor([[0, 1], [1, 2]])
    state = process.corrupt(tokens, schedule(torch.zeros(2)))
    torch.testing.assert_close(state, process.encode(tokens))
    mask = torch.tensor([[False, True], [False, True]])
    state = process.corrupt(tokens, schedule(torch.ones(2)), mask)
    dropped = process.drop_condition(state, mask, 1.0)
    assert torch.isneginf(dropped[:, 0]).all()
    torch.testing.assert_close(state[:, 1], dropped[:, 1])


def test_expected_and_argmax_input_and_dropped_prompt():
    for mode in ("expectation", "argmax"):
        model = instantiate(tiny_config(**{"model.input_mode": mode}).algorithm).model
        logp = torch.tensor([[[0.2, 0.8] + [0.0] * 10]], dtype=torch.float64).log()
        embedded = model.embed_state(logp)
        table = model.vocab_embed.embedding
        expected = (
            0.2 * table[0] + 0.8 * table[1] if mode == "expectation" else table[1]
        )
        torch.testing.assert_close(embedded[0, 0], expected)
        assert not model.embed_state(torch.full_like(logp, -torch.inf)).any()


def test_training_bootstrap_shares_state_time_and_detaches_feedback():
    model = instantiate(
        tiny_config(True, **{"self_conditioning.probability": 1.0}).algorithm
    )
    calls = []
    hook = model.model.register_forward_hook(
        lambda m, args, kwargs, output: calls.append((args, kwargs, output)),
        with_kwargs=True,
    )
    tokens = torch.randint(0, 12, (3, 6))
    model.loss(tokens, torch.ones_like(tokens), training=True)["loss"].backward()
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0][0], calls[1][0][0])
    torch.testing.assert_close(calls[0][0][1], calls[1][0][1])
    assert not calls[1][1]["self_condition"].requires_grad
    hook.remove()


@pytest.mark.parametrize(
    "preset",
    [
        "sudoku_simplex",
        "sudoku_simplex_argmax",
        "sudoku_simplex_adaptive",
        "sudoku_simplex_self_conditioning",
    ],
)
def test_presets_compose(preset):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
        version_base=None,
    ):
        config = compose(
            config_name="config",
            overrides=[
                f"+experiment={preset}",
                "model.hidden_size=16",
                "model.n_blocks=1",
                "model.n_heads=2",
            ],
        )
    assert instantiate(config.algorithm).model.vocab_size == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sc", [False, True])
@pytest.mark.parametrize("input_mode", ["expectation", "argmax"])
def test_cuda_loss_backward_and_sampling(sc, input_mode):
    model = instantiate(
        tiny_config(sc, **{"model.input_mode": input_mode}).algorithm
    ).cuda()
    x = torch.randint(1, 10, (2, 6), device="cuda")
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :3] = False
    loss = model.loss(x, mask, training=True)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.output_layer.linear.weight.grad.norm() > 0
    model.eval()
    samples, metadata = model.generate(
        SimplexBridgeSampler(steps=3, churn=0.2), x, ~mask
    )
    assert metadata["nfe"] == 3
    assert torch.equal(samples[:, :3], x[:, :3])
