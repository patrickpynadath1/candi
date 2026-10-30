"""LangFlow quantile/reference parity, schedule fitting and recipe integration."""

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.noise_schedules.gumbel import GumbelSchedule


def test_quantiles_match_released_gumbel_proposal():
    schedule = GumbelSchedule()
    time = torch.tensor([0.0, 1e-7, 0.01, 0.2, 0.5, 0.9, 1.0], dtype=torch.float64)
    reference = torch.distributions.Gumbel(schedule.loc, schedule.scale).icdf(time)
    lower = schedule.loc - math.log(-math.log(schedule.cutoff)) * schedule.scale
    upper = schedule.loc - math.log(schedule.cutoff) * schedule.scale
    reference = reference.clamp(lower, upper)
    gamma, _ = schedule.gamma(time)
    torch.testing.assert_close(gamma, reference)


def test_path_derivatives_match_autograd_and_vp_identity():
    schedule = GumbelSchedule()
    time = torch.linspace(0.001, 0.999, 101, dtype=torch.float64, requires_grad=True)
    path = schedule(time)
    for value, derivative in [
        (path.alpha, path.dalpha_dt),
        (path.sigma, path.dsigma_dt),
    ]:
        (actual,) = torch.autograd.grad(value.sum(), time, retain_graph=True)
        torch.testing.assert_close(actual, derivative)
    torch.testing.assert_close(
        path.alpha.square() + path.sigma.square(), torch.ones_like(time)
    )
    assert (path.dalpha_dt < 0).all() and (path.dsigma_dt > 0).all()


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
def test_endpoints_are_finite_with_zero_clipped_derivatives(dtype):
    path = GumbelSchedule()(torch.tensor([0.0, 1.0], dtype=dtype))
    for value in (path.alpha, path.sigma, path.dalpha_dt, path.dsigma_dt):
        assert torch.isfinite(value).all()
    assert (path.dalpha_dt == 0).all() and (path.dsigma_dt == 0).all()


def test_profile_fit_improves_and_checkpoint_resumes_identically():
    schedule = GumbelSchedule(loc=0.0, scale=1.0, entropy=4.0, learning_rate=0.02)
    time = torch.linspace(0.001, 0.999, 128, dtype=torch.float64)
    grid = torch.linspace(-2, 5, 128, dtype=torch.float64)

    def target(gamma):
        return 5 * (-(-(gamma - 0.8) / 0.7).exp()).exp()

    def error():
        predicted = (
            schedule.entropy * (-(-(grid - schedule.loc) / schedule.scale).exp()).exp()
        )
        return (predicted - target(grid)).square().mean()

    before = error()
    for step in range(400):
        gamma, _ = schedule.gamma(time)
        schedule.observe(time, target(gamma), step, training=True)
    assert error() < before * 0.05
    restored = GumbelSchedule(learning_rate=0.02)
    restored.load_state_dict(schedule.state_dict())
    losses = target(schedule.gamma(time)[0])
    schedule.observe(time, losses, 401, training=True)
    restored.observe(time, losses, 401, training=True)
    for key, value in schedule.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_detached_feedback_eval_and_empty_batch_do_not_update():
    schedule = GumbelSchedule()
    time = torch.tensor([0.2, 0.8], requires_grad=True)
    loss = torch.tensor([1.0, 2.0], requires_grad=True)
    schedule.observe(time, loss, 0, training=True)
    assert time.grad is None and loss.grad is None
    saved = schedule.fit_parameters.clone()
    schedule.observe(time, loss, 1, training=False)
    schedule.observe(time[:0], loss[:0], 1, training=True)
    schedule.eval()
    schedule.observe(time, loss, 1, training=True)
    torch.testing.assert_close(schedule.fit_parameters, saved)
    schedule.train()
    schedule.learnable = False
    schedule.observe(time, loss, 1, training=True)
    torch.testing.assert_close(schedule.fit_parameters, saved)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0},
        {"cutoff": 0},
        {"cutoff": 1e-30},
        {"entropy": -1},
        {"loc": float("nan")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GumbelSchedule(**kwargs)


@pytest.mark.parametrize("hybrid", [False, True])
def test_schedule_composes_trains_and_generates(hybrid):
    overrides = (
        [
            "+experiment=corruption_factorial",
            "hybrid_view@model.views.content=gaussian_one_hot",
            "noise_schedule=gumbel",
            "model.views.content.schedule=${noise_schedule}",
            "model.backbone.hidden_size=16",
            "model.backbone.n_blocks=1",
            "model.backbone.n_heads=2",
            "model.backbone.cond_dim=8",
        ]
        if hybrid
        else [
            "+experiment=gaussian_base",
            "noise_schedule=gumbel",
            "model.hidden_size=16",
            "model.n_blocks=1",
            "model.n_heads=2",
            "model.cond_dim=8",
        ]
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(
            config_name="config", overrides=overrides + ["sampler.steps=2"]
        )
    algo, sampler = instantiate(config.algorithm), instantiate(config.sampler)
    schedule = algo.views[0].schedule if hybrid else algo.schedule
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    mask = torch.tensor([[False, True, True, True]] * 2)
    result = algo.loss(tokens, mask, training=True)
    result["loss"].backward()
    assert schedule.fit_steps == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in algo.parameters())
    output, metadata = algo.generate(sampler, tokens, ~mask)
    assert torch.equal(output[~mask], tokens[~mask])
    assert torch.isfinite(result["loss"]) and metadata["nfe"] == 3
    assert schedule.fit_steps == 1
