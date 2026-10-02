"""Adaptive fit mathematics, clock derivatives and persisted observation state."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from core.noise_schedules.adaptive import (
    SFLMAdaptiveSchedule,
    CDCDAdaptiveSchedule,
    SFLMGaussianSchedule,
    CDCDGaussianSchedule,
)
from core.noise_schedules.log_linear import LogLinear
from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.noise_schedules.gaussian_rank import OneHotRankSchedule


def make(cls=SFLMAdaptiveSchedule, **kwargs):
    return cls(
        LogLinear(),
        buffer_size=32,
        refit_every=1,
        n_grid=101,
        n_knots=6,
        warmup_steps=0,
        ema=0.5,
        **kwargs,
    )


def observations():
    t = torch.linspace(0.01, 0.99, 32, dtype=torch.float64)
    return t, t.square() + 0.1


def test_sflm_reference_spline_and_legacy_coordinate():
    schedule = make()
    t, loss = observations()
    schedule.observe(t, loss, 0, training=True)
    grid = np.linspace(0, 1, 101)
    model = make_pipeline(
        SplineTransformer(n_knots=6, degree=3, extrapolation="continue"),
        Ridge(alpha=0.001),
    )
    model.fit(t.numpy()[:, None], loss.numpy())
    profile = model.predict(grid[:, None])
    importance = 0.999 * np.maximum(np.gradient(profile, grid), 0) + 0.001
    cdf = np.cumsum(importance)
    cdf /= cdf[-1]
    progress = PchipInterpolator(cdf, grid)(grid)
    keep = 0.001 + 0.999 * (1 - progress)
    expected = PchipInterpolator(grid, keep)
    torch.testing.assert_close(
        schedule.alpha_t(t), torch.from_numpy(expected(t.numpy()))
    )
    torch.testing.assert_close(
        schedule.alpha_prime_t(t), torch.from_numpy(expected.derivative()(t.numpy()))
    )
    # S-FLM buffers the sampled clock, even after its meaning changes on refit.
    schedule.observe(t, loss, 1, training=True)
    torch.testing.assert_close(schedule.t_buf, t)


@pytest.mark.parametrize("cls", [SFLMAdaptiveSchedule, CDCDAdaptiveSchedule])
def test_physical_endpoints_derivative_and_coordinate(cls):
    schedule = make(cls, coordinate="physical")
    t, loss = observations()
    schedule.observe(t, loss, 0, training=True)
    grid = torch.linspace(0, 1, 301, dtype=torch.float64)
    keep, derivative = schedule(grid)
    assert keep[0] == 1 and keep[-1] == pytest.approx(0.001)
    assert torch.all(keep[1:] <= keep[:-1]) and torch.all(derivative <= 0)
    t = t + 0.000123  # Probe between spline knots (PCHIP is only C1 at knots).
    eps = 1e-6
    finite_difference = (schedule(t + eps)[0] - schedule(t - eps)[0]) / (2 * eps)
    torch.testing.assert_close(schedule(t)[1], finite_difference, atol=1e-7, rtol=1e-6)
    expected_progress = (1 - schedule(t)[0]) / 0.999
    schedule.observe(t, loss, 1, training=True)
    torch.testing.assert_close(schedule.t_buf, expected_progress)
    if cls is CDCDAdaptiveSchedule:
        assert torch.isfinite(schedule.importance_buf).all()
        assert not torch.allclose(schedule.importance_buf, torch.ones_like(t))


@pytest.mark.parametrize("cls", [SFLMAdaptiveSchedule, CDCDAdaptiveSchedule])
def test_checkpoint_nested_resume_eval_warmup_and_large_batches(cls):
    schedule = make(cls)
    t, loss = observations()
    schedule.observe(t, loss, 0)
    schedule.eval()
    schedule.observe(t, loss, 0, training=True)
    assert schedule.num_observed == 0
    schedule.train()
    schedule.observe(t.repeat(3), loss.repeat(3), 0, training=True)
    assert schedule.refit_count == 1 and schedule.num_observed == 32
    parent = torch.nn.ModuleDict({"schedule": schedule})
    restored = torch.nn.ModuleDict({"schedule": make(cls)})
    restored.load_state_dict(parent.state_dict())
    torch.testing.assert_close(restored["schedule"](t), schedule(t))
    for item in (schedule, restored["schedule"]):
        item.observe(t, loss * 1.1, 1, training=True)
    torch.testing.assert_close(restored["schedule"](t), schedule(t))
    schedule.warmup_steps = 10
    schedule.observe(t, loss, 2, training=True)
    assert schedule.refit_count == 2


@pytest.mark.parametrize("cls", [SFLMGaussianSchedule, CDCDGaussianSchedule])
def test_gaussian_chain_rule_representation_and_training_clock(cls):
    schedule = cls(
        LinearInterpolationSchedule(),
        buffer_size=32,
        warmup_steps=0,
        refit_every=1,
        n_grid=101,
        coordinate="physical",
    )
    t, loss = observations()
    schedule.observe(t, loss, 0, training=True)
    t = t + 0.000123  # Probe between spline knots (PCHIP is only C1 at knots).
    eps = 1e-6
    path = schedule(t)
    left, right = schedule(t - eps), schedule(t + eps)
    torch.testing.assert_close(
        path.dalpha_dt, (right.alpha - left.alpha) / (2 * eps), atol=1e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        path.dsigma_dt, (right.sigma - left.sigma) / (2 * eps), atol=1e-7, rtol=1e-6
    )
    end = schedule(torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(end.alpha, torch.tensor([1.0, 0.0], dtype=torch.float64))
    rank = OneHotRankSchedule(12, num_points=32, training_epsilon=0.03)
    wrapped = cls(rank)
    assert wrapped.representation_requirements == rank.representation_requirements
    torch.testing.assert_close(wrapped.training_time(t), 0.97 * t)
    torch.testing.assert_close(wrapped(t).sigma, rank(t).sigma)


@pytest.mark.parametrize(
    "name",
    [
        "sflm_adaptive",
        "cdcd_adaptive",
        "sflm_gaussian_adaptive",
        "cdcd_gaussian_adaptive",
    ],
)
def test_hydra_schedule_configs(name):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(config_name="config", overrides=[f"noise_schedule={name}"])
        schedule = instantiate(config.noise_schedule)
    assert schedule.loss_signal == (
        "cross_entropy" if name.startswith("cdcd") else "objective"
    )
    if "gaussian" not in name:
        assert schedule.coordinate == (
            "physical" if name.startswith("cdcd") else "legacy_time"
        )


def test_legacy_boundary_stabilization_and_invalid_clock():
    schedule = make()
    t, loss = observations()
    schedule.observe(t, loss, 0, training=True)
    endpoints = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert schedule.alpha_t(endpoints)[0] > 1  # Reference cumsum extrapolation.
    keep, derivative = schedule(endpoints)
    assert keep[0] == 1 and derivative[0] == 0
    from core.objectives.weighted_ce import WeightedCrossEntropy

    near_zero = torch.tensor([0.001], dtype=torch.float32)
    noisy_keep, noisy_derivative = schedule(near_zero)
    assert noisy_keep.item() < 1
    objective = WeightedCrossEntropy()
    value = objective(
        torch.zeros(1, 1, 2).log_softmax(-1),
        torch.zeros(1, 1, dtype=torch.long),
        noisy_keep,
        noisy_derivative,
    )
    assert torch.isfinite(value).all()
    for invalid in (float("nan"), -0.1, 1.1):
        with pytest.raises(ValueError, match="schedule time"):
            schedule(torch.tensor([invalid]))


def test_cdcd_flat_loss_fallback_and_finite_buffer_filter():
    schedule = make(CDCDAdaptiveSchedule)
    t, _ = observations()
    schedule.observe(
        torch.cat([t, torch.tensor([float("nan")])]), torch.ones(33), 0, training=True
    )
    assert schedule.num_observed == 32
    torch.testing.assert_close(schedule(t), LogLinear()(t), atol=1e-12, rtol=1e-12)
    schedule.observe(t[:0], t[:0], 1, training=True)
    assert schedule.refit_count == 1


def test_gaussian_base_is_a_replaceable_hydra_group():
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                "noise_schedule=sflm_gaussian_adaptive",
                "noise_schedule@noise_schedule.base=linear_interpolation",
            ],
        )
    schedule = instantiate(config.noise_schedule)
    assert isinstance(schedule.base, LinearInterpolationSchedule)
