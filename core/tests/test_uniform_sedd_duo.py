"""Discrete process mathematics, distinct parameterizations and DUO curriculum."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from hydra.errors import InstantiationException
from scipy.integrate import quad
from scipy.stats import norm

from core.parameterizations.plug_in import PlugInParameterization
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.score import AbsorbingScoreParameterization
from core.processes.uniform import UniformProcess
from core.processes.absorbing import AbsorbingProcess
from core.objectives.uniform import UniformDiffusionObjective
from core.objectives.score_entropy import AbsorbingScoreEntropy
from core.noise_schedules.duo_calibration import (
    CachedIntegral,
    DuoCalibration,
    build_integral_table,
)
from core.curricula.duo import DuoCurriculum
from core.models.layers import EmbeddingLayer
from core.models.continuous_dit import ContinuousDiT
from core.samplers.uniform_ancestral import UniformAncestralSampler
from core.samplers.uniform_entropy_bounded import UniformEntropyBoundedSampler
from core.samplers.analytic import AnalyticSampler
from core.samplers.ancestral import AncestralSampler
from core.training.module import TrainingModule


@pytest.fixture(scope="module")
def table_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("duo") / "v12.npz"
    np.savez(path, **build_integral_table(12, num_points=10001))
    return str(path)


def config(method, table_path=None, *overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        args = [
            f"+experiment={method}",
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=1",
            "model.cond_dim=16",
            "model.dropout=0",
            "sampler.steps=3",
            "output_dir=/tmp/core_discrete_test",
            "data.cache_dir=/tmp/core_discrete_data",
            "algorithm.length=10",
            "loader.global_batch_size=2",
            "trainer.accumulate_grad_batches=1",
        ]
        if method == "duo":
            args += [
                f"calibration.cache_path={table_path}",
                "curriculum.start_step=1",
                "curriculum.end_step=2",
            ]
        return compose(config_name="config", overrides=[*args, *overrides])


def test_plugin_not_posterior_and_normalization_isolated():
    logits = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    predictor = PlugInParameterization(temperature=0.2, top_k=1)
    assert not isinstance(predictor, PosteriorParameterization)
    torch.testing.assert_close(
        predictor.training_plugin(logits), logits.log_softmax(-1)
    )
    assert predictor.sampling_plugin(logits).exp().tolist() == [[[0.0, 0.0, 1.0]]]
    assert PlugInParameterization(commitment="greedy").probabilities(
        logits.log_softmax(-1), apply_commitment=True
    ).tolist() == [[[0.0, 0.0, 1.0]]]
    (-predictor.training_plugin(logits)[..., 0]).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_duo_input_softmax_and_gaussian_input_linear_projection():
    layer = EmbeddingLayer(2, 3)
    scores = torch.tensor([[[1.0, -1.0, 2.0]]])
    torch.testing.assert_close(layer(scores), scores.softmax(-1) @ layer.embedding)
    assert not torch.allclose(layer(scores), scores @ layer.embedding)
    model = ContinuousDiT(
        3, hidden_size=8, n_heads=2, n_blocks=1, cond_dim=8, dropout=0
    )
    captured = []
    hook = model.input_projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach())
    )
    model(scores, torch.tensor([0.5]), torch.ones(1), torch.ones(1))
    hook.remove()
    expected = (
        scores @ torch.nn.functional.normalize(model.input_embedding, dim=-1) / 2**0.5
    )
    torch.testing.assert_close(captured[0], expected)


def test_uniform_transition_bayes_kernel_and_clean_limit():
    process = UniformProcess(3)
    estimate = torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float64)
    state = torch.tensor([[1]])
    at, ass = torch.tensor([0.3]), torch.tensor([0.8])
    actual = process.reverse_probabilities(state, estimate, at, ass)
    # Evaluate q(xt|xs)*p_theta(xs) and normalize, independently of the expansion.
    likelihood = (at / ass).double()[:, None, None] * torch.nn.functional.one_hot(
        state, 3
    ) + (1 - at / ass).double()[:, None, None] / 3
    marginal = (
        ass.double()[:, None, None] * estimate + (1 - ass).double()[:, None, None] / 3
    )
    reference = likelihood * marginal
    reference /= reference.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        actual.sum(-1), torch.ones((1, 1), dtype=torch.float64), atol=1e-7, rtol=0
    )
    same = process.reverse_probabilities(state, estimate, at, at)
    torch.testing.assert_close(same, torch.nn.functional.one_hot(state, 3).double())
    tokens = torch.zeros((30000, 1), dtype=torch.long)
    torch.manual_seed(5)
    corrupted = process.corrupt(
        tokens, torch.full((len(tokens),), 0.6), torch.ones_like(tokens)
    )
    assert (corrupted == 0).float().mean().item() == pytest.approx(
        0.6 + 0.4 / 3, abs=0.01
    )


def test_exact_estimates_have_zero_objectives_and_scores_are_not_softmax():
    x = torch.tensor([[0, 1]])
    state = torch.tensor([[0, 2]])
    log_weights = torch.nn.functional.one_hot(x, 3).float().log()
    loss = UniformDiffusionObjective(3)(
        log_weights, state, x, torch.tensor([[0.5]]), torch.tensor([[-1.0]])
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    keep = torch.tensor([[0.6]])
    mask = 3
    state = torch.tensor([[3, 1]])
    log_scores = torch.full((1, 2, 4), -torch.inf)
    log_scores[0, 0, 0] = (keep / (1 - keep)).log()
    log_scores.scatter_(-1, state[..., None], 0)
    loss = AbsorbingScoreEntropy(mask, 4)(
        log_scores, state, x, keep, torch.tensor([[-1.0]])
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    scores = AbsorbingScoreParameterization()(
        torch.zeros(1, 2, 4), state, -keep[:, 0].log()
    )
    assert torch.equal(scores.gather(-1, state[..., None]), torch.zeros(1, 2, 1))
    assert not torch.allclose(scores.exp().sum(-1), torch.ones(1, 2))


def test_absorbing_analytic_oracle_probabilities():
    process = AbsorbingProcess(3)
    state = torch.tensor([[3, 1]])
    keep_t, keep_s = torch.tensor([0.2]), torch.tensor([0.6])
    scores = torch.zeros(1, 2, 4, dtype=torch.float64)
    scores[0, 0, 2] = (keep_t / (1 - keep_t)).item()
    scores[0, 0, 3] = 1
    scores[0, 1, 1] = 1
    p = process.analytic_probabilities(state, scores, keep_s.log() - keep_t.log())
    expected = torch.tensor(
        [[[0.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float64
    )
    torch.testing.assert_close(p, expected, atol=1e-7, rtol=1e-6)
    final = process.analytic_probabilities(state, scores, -keep_t.log(), denoise=True)
    assert torch.equal(final.argmax(-1), torch.tensor([[2, 1]]))


@pytest.mark.parametrize("method", ["uniform_diffusion", "sedd_absorb", "duo"])
def test_method_configs_loss_gradients_and_checkpoint(method, table_path, tmp_path):
    c = config(method, table_path)
    a = instantiate(c.algorithm)
    tokens = torch.randint(0, 12, (2, 10))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :3] = False
    result = a.loss(tokens, mask, training=True, step=0)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(
        torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None
    )
    assert a.model.adaLN
    samplers = (
        [AnalyticSampler(3)]
        if method == "sedd_absorb"
        else [UniformAncestralSampler(3), UniformEntropyBoundedSampler(steps=10)]
    )
    for sampler in samplers:
        sample, meta = a.generate(sampler, tokens, ~mask)
        assert torch.equal(sample[~mask], tokens[~mask])
        assert ((sample >= 0) & (sample < 12)).all()
        assert meta["nfe"] <= 11
    path = tmp_path / "weights.pt"
    torch.save(a.state_dict(), path)
    b = instantiate(c.algorithm)
    b.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.manual_seed(7)
    v1 = a.loss(tokens, mask, training=False)["loss"]
    torch.manual_seed(7)
    v2 = b.loss(tokens, mask, training=False)["loss"]
    torch.testing.assert_close(v1, v2, rtol=0, atol=0)


def test_compatible_predictors_enforced(table_path):
    with pytest.raises(InstantiationException, match="PlugInParameterization"):
        instantiate(
            config(
                "uniform_diffusion", table_path, "parameterization=posterior"
            ).algorithm
        )
    with pytest.raises(InstantiationException, match="AbsorbingScoreParameterization"):
        instantiate(
            config("sedd_absorb", table_path, "parameterization=plug_in").algorithm
        )
    a = instantiate(config("sedd_absorb", table_path).algorithm)
    tokens = torch.zeros((1, a.length), dtype=torch.long)
    with pytest.raises(TypeError, match="PosteriorParameterization"):
        a.generate(
            AncestralSampler(2), tokens, torch.zeros_like(tokens, dtype=torch.bool)
        )


def test_duo_curriculum_boundaries_validation_and_trainer_context(table_path):
    a = instantiate(config("duo", table_path).algorithm)
    tokens = torch.randint(0, 12, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    observed = []
    hook = a.model.vocab_embed.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    for step, training, dimension in [
        (0, True, 3),
        (1, True, 3),
        (2, True, 3),
        (3, True, 2),
        (0, False, 2),
    ]:
        torch.manual_seed(3)
        loss = a.loss(tokens, mask, step=step, training=training)["loss"]
        assert torch.isfinite(loss)
        assert observed[-1].ndim == dimension
    hook.remove()
    assert a.curriculum.inverse_temperature(2) == 1e10
    curriculum = DuoCurriculum(
        start_step=10, end_step=20, log10_temperature_start=1, log10_temperature_end=-1
    )
    assert [curriculum.inverse_temperature(step) for step in (0, 15, 30)] == [
        0.1,
        1.0,
        1e10,
    ]
    module = TrainingModule(config("duo", table_path))
    module._trainer = SimpleNamespace(
        global_step=2, world_size=1, global_rank=0, accumulate_grad_batches=1
    )
    module.log = lambda *args, **kwargs: None
    contexts = []

    def loss(*args, **kwargs):
        contexts.append(kwargs)
        return {"loss": torch.tensor(1.0)}

    module.algorithm.loss = loss
    module.training_step({"input_ids": tokens, "loss_mask": mask}, 0)
    assert contexts[-1]["step"] == 2 and contexts[-1]["training"] is True


def test_integral_lookup_derivative_and_vocab_guard(table_path, tmp_path):
    gamma = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    values = torch.tensor([0.1, 0.5, 0.9])
    derivatives = torch.tensor([1.0, 2.0, 3.0])
    result = CachedIntegral.apply(gamma, values, derivatives, 0.0, 1.0)
    torch.testing.assert_close(result, values)
    result.sum().backward()
    torch.testing.assert_close(gamma.grad, derivatives)
    with pytest.raises(ValueError, match="vocabulary"):
        DuoCalibration(13, cache_path=table_path)
    calibration = DuoCalibration(12, cache_path=table_path)
    keep = torch.tensor([0.2, 0.8])
    derivative = torch.tensor([-0.999, -0.999])
    noise = calibration(keep, derivative)
    expected = (
        1.75
        * 0.999
        * 1000
        * (
            calibration.keep_from_gamma(noise.gamma + 0.001)
            - calibration.keep_from_gamma(noise.gamma)
        )
    )
    torch.testing.assert_close(noise.derivative, expected)
    assert (noise.derivative <= 0).all()


def test_integral_generation_matches_reference_quadrature():
    table = build_integral_table(12, gamma_min=-3.5, gamma_max=-1.75, num_points=3)
    for i, gamma in enumerate(np.linspace(-3.5, -1.75, 3)):
        shift = np.exp(-gamma / 2)

        def density(z):
            return norm.pdf(z, loc=shift) * norm.cdf(z) ** 11

        probability = quad(density, -np.inf, np.inf)[0]
        derivative = quad(
            lambda z: -0.5 * shift * (z - shift) * density(z), -np.inf, np.inf
        )[0]
        assert table["pt"][i] == pytest.approx(probability, abs=1e-9)
        assert table["grad_pt"][i] == pytest.approx(derivative, abs=1e-9)


def test_composed_recipes_retain_reference_defaults():
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        for method in ("uniform_diffusion", "sedd_absorb", "duo"):
            c = compose(config_name="config", overrides=[f"+experiment={method}"])
            assert (
                c.model.hidden_size,
                c.model.n_blocks,
                c.model.n_heads,
                c.model.adaLN,
            ) == (768, 12, 12, True)
            assert c.trainer.max_steps == 1000000 and c.loader.global_batch_size == 512
            assert c.sampler.steps == 1024 and c.sampler.epsilon == 0.001
            assert c.sampler.final_denoise == "ancestral" and c.sampler.use_float64
            if method == "duo":
                assert (c.curriculum.start_step, c.curriculum.end_step) == (
                    100000,
                    200000,
                )
                assert (
                    c.curriculum.log10_temperature_start,
                    c.curriculum.log10_temperature_end,
                ) == (-1.0, -2.0)
                assert (c.calibration.gamma_min, c.calibration.gamma_max) == (
                    -3.5,
                    -1.75,
                )
