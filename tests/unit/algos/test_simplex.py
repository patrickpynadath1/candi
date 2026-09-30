"""Equation-level SDM checks, independent analytic moments and boundary cases."""

import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

import main  # noqa: F401
from algos.simplex import SimplexDiffusion
from algos.simplex_utils import (
    concentration,
    corruption_parameters,
    log_dirichlet_sample,
    reverse_bridge,
    guided_log_probs,
    dmd_generator_loss,
)
from algos.paper_time_sampler import PaperTimeSampler
from dataloader import get_tokenizer
from samplers import create_sampler, run_sampler, SamplingRequest


def tiny_config(sc=False, **overrides):
    entries = [
        "+experiment=sudoku_simplex",
        "model.hidden_size=16",
        "model.n_heads=2",
        "model.n_blocks=1",
        "model.cond_dim=8",
        "model.length=6",
        "model.dropout=0",
        "model.output_dropout=0",
        "training.ema=0",
        "trainer.devices=1",
        f"self_conditioning.enabled={str(sc).lower()}",
    ]
    entries.extend(f"{key}={value}" for key, value in overrides.items())
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        return compose(config_name="config", overrides=entries)


@pytest.mark.parametrize("disable_ema", [False, True])
def test_training_evaluation_respects_current_weight_override(disable_ema):
    config = tiny_config(
        **{
            "training.ema": 0.9999,
            "eval.disable_ema": str(disable_ema).lower(),
        }
    )
    model = SimplexDiffusion(config, get_tokenizer(config))
    parameter = next(model._get_parameters())
    with torch.no_grad():
        parameter.fill_(2)
        model.ema.shadow_params[0].fill_(5)
    model._eval_mode()
    torch.testing.assert_close(
        parameter, torch.full_like(parameter, 2 if disable_ema else 5)
    )
    assert not model.backbone.training
    model._train_mode()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 2))
    assert model.backbone.training
    assert not model.ema.collected_params


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


@pytest.mark.parametrize("churn", [0.0, 0.2, 1.0])
def test_full_sampler_recovers_known_data_distribution(churn):
    """Exact Bayes denoiser checks the entire chain, not just one bridge.

    With uniform reference, all class-conditional Dirichlet normalizers
    are equal. Bayes logits are log(data prior) + c(t)(1-t) log(P_t).
    Every reverse step must preserve this known mixture marginal.
    """
    torch.manual_seed(27)
    config = tiny_config(**{"model.length": 1})
    model = SimplexDiffusion(config, get_tokenizer(config)).eval()
    data_prior = torch.full((model.vocab_size,), 0.1 / (model.vocab_size - 2))
    data_prior[:2] = torch.tensor([0.7, 0.2])

    def exact_posterior(log_state, t):
        strength = model.concentration(t.double()) * (1 - t.double())
        return (data_prior.log() + strength[:, None, None] * log_state).log_softmax(-1)

    model.forward = exact_posterior
    sampler = create_sampler(config, sampling_overrides={"churn": churn})
    result = run_sampler(sampler, model, SamplingRequest(2048, num_steps=12))
    frequencies = result.samples.flatten().bincount(minlength=model.vocab_size) / 2048
    torch.testing.assert_close(frequencies, data_prior, atol=0.035, rtol=0)
    assert result.metadata["nfe"] == 12


@pytest.mark.parametrize("sc", [False, True])
@pytest.mark.parametrize("input_mode", ["expectation", "argmax"])
def test_real_loss_gradient_and_prompt_preservation(sc, input_mode):
    c = tiny_config(sc, **{"algo.input_mode": input_mode})
    m = SimplexDiffusion(c, get_tokenizer(c))
    x = torch.randint(1, 10, (2, 6))
    mask = torch.ones_like(x)
    mask[:, :3] = 0
    loss = m._loss(x, mask).loss
    loss.backward()
    grad = m.backbone.output_layer.linear.weight.grad
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm() > 0
    m.eval()
    for churn in (0.0, 0.2, 1.0):
        sampler = create_sampler(c, sampling_overrides={"churn": churn})
        result = run_sampler(
            sampler,
            m,
            SamplingRequest(2, num_steps=3, prompt_tokens=x, prompt_mask=~mask.bool()),
        )
        assert result.metadata["nfe"] == 3
        assert torch.equal(result.samples[:, :3], x[:, :3])
    restored = SimplexDiffusion(c, get_tokenizer(c))
    restored.load_state_dict(m.state_dict())


def test_adaptive_clock_concentrates_on_loss_derivative_and_reloads():
    sampler = PaperTimeSampler(
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
    restored = PaperTimeSampler(adaptive=True, buffer_size=2000)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.cdf, sampler.cdf)
    with pytest.raises(ValueError, match="fitted"):
        PaperTimeSampler().sampling_grid(3, "adaptive", "cpu")


def test_guidance_and_dmd_gradient():
    a, b = torch.randn(2, 3).log_softmax(-1), torch.randn(2, 3).log_softmax(-1)
    torch.testing.assert_close(guided_log_probs(a, b, 0), a)
    torch.testing.assert_close(guided_log_probs(a, b, 1), b)
    clean = torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.4], dtype=torch.float64)
    beta = corruption_parameters(clean, t, torch.ones_like(t), torch.full((3,), 1 / 3))
    logp = log_dirichlet_sample(beta)
    auxiliary = torch.tensor([[[0.1, 0.2, 0.7]]], requires_grad=True)
    teacher = torch.tensor([[[0.3, 0.3, 0.4]]], requires_grad=True)
    dmd_generator_loss(logp, auxiliary, teacher, 1 - t).sum().backward()
    assert torch.isfinite(clean.grad).all() and clean.grad.norm() > 0
    assert auxiliary.grad is None and teacher.grad is None


def test_algorithm4_updates_generator_and_auxiliary_not_teacher():
    from algos.simplex_dmd import SimplexDMD

    config = tiny_config(**{"algo": "simplex_dmd"})
    model = SimplexDMD(config, get_tokenizer(config))
    model.log = lambda *args, **kwargs: None
    with torch.no_grad():
        model.auxiliary.output_layer.linear.weight.normal_(0, 0.01)
    x = torch.randint(1, 10, (2, 6))
    mask = torch.ones_like(x)
    mask[:, :3] = 0
    loss = model.nll(x, None, train_mode=True, valid_tokens=mask)
    objective = (loss * mask).sum() / mask.sum()
    assert torch.isfinite(objective)
    objective.backward()
    assert model.backbone.output_layer.linear.weight.grad.norm() > 0
    assert model.auxiliary.output_layer.linear.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.teacher.parameters())


def test_guided_and_confidence_sampling_preserve_condition_and_count_calls():
    config = tiny_config(True, **{"algo.condition_dropout": 0.1})
    model = SimplexDiffusion(config, get_tokenizer(config)).eval()
    x = torch.randint(1, 10, (2, 6))
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :3] = True
    sampler = create_sampler(
        config, sampling_overrides={"guidance_scale": 2.0, "confidence_power": 1.0}
    )
    result = run_sampler(
        sampler, model, SamplingRequest(2, 3, prompt_tokens=x, prompt_mask=mask)
    )
    assert result.metadata["nfe"] == 6
    assert torch.equal(result.samples[:, :3], x[:, :3])
    assert model.backbone.latent_self_conditioning.previous is None


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


def test_paper_ema_uses_fixed_decay_without_repository_warmup():
    config = tiny_config(**{"training.ema": 0.9999})
    model = SimplexDiffusion(config, get_tokenizer(config))
    assert model.ema.decay == 0.9999 and model.ema.num_updates is None
