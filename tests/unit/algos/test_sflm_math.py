import torch

from algos.flm_utils import alpha_to_gamma, build_luts, gamma_to_alpha
from algos.noise_schedules import (
    AdaptiveSchedule,
    CosineSquared,
    LogLinear,
    TruncatedSchedule,
)
from algos.spherical_utils import exp_map, log_map, slerp, sphere_normalize


def test_spherical_maps_round_trip():
    x = sphere_normalize(torch.randn(2, 3, 5))
    y = sphere_normalize(torch.randn(2, 3, 5))
    restored = exp_map(x, log_map(x, y))
    assert torch.allclose(restored, y, atol=2e-5)
    assert torch.allclose(restored.norm(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_slerp_endpoints():
    x = sphere_normalize(torch.randn(2, 4))
    y = sphere_normalize(torch.randn(2, 4))
    assert torch.allclose(slerp(x, y, torch.tensor(0.0)), x, atol=2e-5)
    assert torch.allclose(slerp(x, y, torch.tensor(1.0)), y, atol=2e-5)


def test_flm_luts_are_monotone_and_invertible():
    a2g, g2a = build_luts(8, size=256)
    alpha = torch.linspace(a2g[0, 0], a2g[0, -1], 20)
    reconstructed = gamma_to_alpha(alpha_to_gamma(alpha, a2g), g2a)
    assert torch.all(a2g[0, 1:] >= a2g[0, :-1])
    assert torch.allclose(reconstructed, alpha, atol=2e-3)


def test_noise_schedule_endpoints_and_truncation():
    for schedule in (LogLinear(), CosineSquared()):
        derivative, alpha = schedule(torch.tensor([0.0, 1.0]))
        assert alpha[0] > alpha[1]
        assert torch.all(derivative <= 1e-6)
    trunc = TruncatedSchedule(LogLinear(), 0.2, 0.8)
    assert torch.allclose(
        trunc.alpha_t(torch.tensor([0.0, 1.0])), torch.tensor([0.8, 0.2]), atol=1e-5
    )


def test_adaptive_schedule_refits_and_restores_spline():
    schedule = AdaptiveSchedule(
        TruncatedSchedule(LogLinear(), 0.001, 0.093),
        buffer_size=8,
        refit_every=1,
        n_grid=32,
        n_knots=4,
        spline_degree=2,
        ridge_alpha=1e-3,
        uniform_mix=1e-3,
        max_steps=10,
        warmup_steps=0,
        ema=0.9,
    )
    t = torch.linspace(0, 1, 8)
    schedule.record_time_loss_pair(t, t.square(), step=1)
    assert schedule.has_schedule
    alpha = schedule.alpha_t(t)
    assert torch.isfinite(alpha).all()
    assert torch.all(alpha[:-1] >= alpha[1:])

    restored = AdaptiveSchedule(
        TruncatedSchedule(LogLinear(), 0.001, 0.093),
        buffer_size=8,
        refit_every=1,
        n_grid=32,
        n_knots=4,
        spline_degree=2,
        ridge_alpha=1e-3,
        uniform_mix=1e-3,
        max_steps=10,
        warmup_steps=0,
        ema=0.9,
    )
    restored.load_state_dict(schedule.state_dict())
    assert torch.allclose(restored.alpha_t(t), alpha)
