import numpy as np
import pytest
import torch

from algos.rank_schedules import (
    SphereRankSchedule, embedding_table_rank, geometric_rank, onehot_rank,
    sigma_from_rank,
)
from algos.noise_schedules import AdaptiveSchedule, LogLinear


def adaptive(**kwargs):
    options = dict(base=LogLinear(0), buffer_size=256, refit_every=1,
                   n_grid=1001, n_knots=12, spline_degree=3, ridge_alpha=1e-8,
                   uniform_mix=1e-4, max_steps=100, warmup_steps=0, ema=0,
                   coordinate="physical")
    options.update(kwargs)
    return AdaptiveSchedule(**options)


def test_rank_inverse_reaches_point_49_without_vocab_scaling():
    rank = geometric_rank(torch.linspace(0, 1, 101, dtype=torch.float64))
    sigma = sigma_from_rank(rank)
    assert sigma[-1].item() == pytest.approx(28.2065247473)
    assert torch.allclose(onehot_rank(sigma), rank, atol=1e-12)
    assert torch.all(sigma.diff() > 0)
    for bad in (0., .5, 1., float("nan")):
        with pytest.raises(ValueError):
            sigma_from_rank(torch.tensor(bad))


def test_onehot_rank_matches_empirical_fraction_of_competitors():
    noise = torch.randn(100_000, 12) * 2
    noise[:, 0] += 1
    rank = (noise[:, 1:] > noise[:, :1]).float().mean()
    assert rank.item() == pytest.approx(onehot_rank(torch.tensor(2.)).item(), abs=.004)


def test_embedding_rank_formula_matches_correlated_ranking_events():
    e = torch.nn.functional.normalize(torch.randn(12, 16), dim=-1)
    indices = torch.randint(12, (50_000,))
    z = .5 * e[indices] + .5 * torch.randn(50_000, 16)
    score = z @ e.T
    observed = (score > score[torch.arange(len(indices)), indices, None]).sum(-1).float().mean()/11
    assert observed.item() == pytest.approx(embedding_table_rank(e, .5).item(), abs=.004)


@pytest.mark.parametrize("dimension", [2, 16, 512])
def test_sphere_inverse_and_endpoint_bridges(dimension):
    schedule = SphereRankSchedule(dimension)
    s = torch.linspace(0, 1, 1001, dtype=torch.float64)
    u = schedule(s)
    assert u[0] == 0 and u[-1] == 1
    assert torch.all(u.diff() > 0)
    assert torch.isfinite(u).all()
    # Independent denser quadrature / table validates inversion accuracy.
    reference = SphereRankSchedule(dimension, grid_size=16385, quadrature_order=512)
    target = geometric_rank(torch.linspace(0, 1, 100, dtype=torch.float64))
    assert torch.allclose(schedule.inverse(target), reference.inverse(target), atol=3e-5)


@pytest.mark.parametrize("coupling", ["linear", "geometric"])
def test_gaussian_embedding_rank_can_preserve_finite_rank_endpoints(coupling):
    schedule = SphereRankSchedule(512, endpoint_width=0)
    progress = torch.tensor([0.0, 1.0], dtype=torch.float64)
    interpolation = schedule(progress, coupling=coupling)
    rank = schedule.ranks.new_tensor([schedule.rank_min, schedule.rank_max])

    assert torch.allclose(interpolation, schedule.inverse(rank), atol=1e-12)
    assert torch.all(interpolation > 0)
    assert torch.all(interpolation < 1)


def test_adaptive_schedule_linearizes_quadratic_loss_and_keeps_endpoints():
    schedule = adaptive()
    s = torch.linspace(0, 1, 256, dtype=torch.float64)
    schedule.record_time_loss_pair(s, s.square(), 0)
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    progress = 1-schedule.alpha_t(t)
    assert progress[0] == 0 and progress[-1] == 1
    assert torch.all(progress.diff() > 0)
    assert torch.allclose(progress.square(), t, atol=.012)
    # Refit using the new clock: observations must stay in physical progress.
    next_s = 1-schedule.alpha_t(s)
    schedule.record_time_loss_pair(s, next_s.square(), 1)
    assert torch.allclose((1-schedule.alpha_t(t)).square(), t, atol=.012)
    assert np.isfinite(schedule.snapshot()["warped_loss_profile"]).all()


def test_adaptive_nested_checkpoint_reload_invalidates_cached_spline():
    parent = torch.nn.Module()
    parent.schedule = adaptive()
    s = torch.linspace(0, 1, 256, dtype=torch.float64)
    parent.schedule.record_time_loss_pair(s, s.square(), 0)
    state = {k: v.clone() for k, v in parent.state_dict().items()}
    expected = parent.schedule.alpha_t(s).clone()
    parent.schedule.record_time_loss_pair(s, torch.ones_like(s), 1)
    parent.load_state_dict(state)
    assert torch.equal(parent.schedule.alpha_t(s), expected)


def test_adaptive_buffer_tracks_fullness_between_refit_steps_and_large_batches():
    schedule = adaptive(buffer_size=8, refit_every=3)
    t = torch.linspace(0, 1, 20, dtype=torch.float64)
    schedule.record_time_loss_pair(t, t, 1)
    assert not schedule.has_schedule
    schedule.record_time_loss_pair(t[:1], t[:1], 3)
    assert schedule.has_schedule


@pytest.mark.parametrize('dimension', [2, 64, 512, 4096])
def test_slerp_rank_dimension_and_inversion(dimension):
    from algos.rank_schedules import SLERPRankSchedule
    schedule = SLERPRankSchedule(dimension)
    rho = torch.linspace(0, 1, 101, dtype=torch.float64)
    u = schedule(rho)
    assert torch.isfinite(schedule.ranks).all()
    if dimension == 4096:
        assert .96 < u[0] < .97
    assert torch.all(u.diff() > 0)
    assert torch.allclose(schedule.expected_rank(u), .01*49**rho, atol=1e-8)
    if dimension == 2:
        assert torch.allclose(u, 2*.01*49**rho, atol=1e-8)
    if dimension == 512:
        assert u[0].item() == pytest.approx(.911375230, abs=2e-7)
        assert u[-1].item() == pytest.approx(.999001819, abs=2e-7)
    restored = SLERPRankSchedule(dimension)
    restored.load_state_dict(schedule.state_dict())
    assert torch.equal(restored(rho), u)


def test_slerp_rank_exact_endpoint_option():
    from algos.rank_schedules import SLERPRankSchedule
    schedule = SLERPRankSchedule(64, endpoint_width=.01)
    u = schedule(torch.tensor([0., 1.]))
    assert torch.allclose(u, torch.tensor([0., 1.]))


def test_embedding_linear_rank_differs_from_linear_interpolation_and_geometric():
    from algos.rank_schedules import SphereRankSchedule
    schedule = SphereRankSchedule(512)
    rho = torch.linspace(.01, .99, 21, dtype=torch.float64)
    target = .01 + (.49-.01)*(rho-.01)/.98
    u = schedule(rho, coupling='linear')
    assert torch.allclose(u, schedule.inverse(target), atol=1e-12)
    assert not torch.allclose(u, rho)
    assert not torch.allclose(u, schedule(rho))
