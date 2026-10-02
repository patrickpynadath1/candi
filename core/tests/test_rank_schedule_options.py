"""Gaussian rank normalization/inversion and VE path derivatives."""

import pytest
import torch
from hydra.utils import instantiate

from core.noise_schedules.gaussian_rank import OneHotRankSchedule
from core.noise_schedules.exponential_ve import ExponentialVESchedule
from core.noise_schedules.adaptive import CDCDGaussianSchedule
from core.noise_schedules.utils import expected_rank, sigma_from_time_vectorized
from core.tests.test_uniform_sedd_duo import config


@pytest.mark.parametrize("inversion", ["lookup", "analytic"])
@pytest.mark.parametrize("coupling", ["linear", "geometric"])
@pytest.mark.parametrize("normalization", ["vocabulary", "competitors"])
def test_rank_roundtrip_and_derivative(inversion, coupling, normalization):
    schedule = OneHotRankSchedule(
        12,
        inversion=inversion,
        coupling=coupling,
        normalization=normalization,
        num_points=20000,
    )
    time = torch.tensor([0.13713, 0.41414, 0.71717], dtype=torch.float64)
    path = schedule(time)
    rank = torch.special.ndtr(-1 / (2**0.5 * path.sigma))
    if normalization == "vocabulary":
        rank *= 11 / 12
    target = 0.01 + time * 0.39 if coupling == "linear" else 0.01 * 40**time
    torch.testing.assert_close(rank, target, atol=2e-7, rtol=2e-5)
    delta = 1e-8
    finite_difference = (
        schedule(time + delta).sigma - schedule(time - delta).sigma
    ) / (2 * delta)
    torch.testing.assert_close(path.dsigma_dt, finite_difference, atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(path.alpha, torch.ones_like(time))


def test_rank_defaults_preserve_legacy_lookup_and_normalization_conversion():
    time = torch.linspace(0, 1, 50)
    legacy = OneHotRankSchedule(12)
    expected = sigma_from_time_vectorized(
        0.01 + 0.39 * time,
        legacy.sigmas,
        expected_rank(12, torch.tensor(1.0), legacy.sigmas),
    )
    torch.testing.assert_close(legacy(time).sigma, expected, rtol=0, atol=0)
    vocab = OneHotRankSchedule(12, inversion="analytic")
    competitors = OneHotRankSchedule(
        12,
        inversion="analytic",
        normalization="competitors",
        min_percentile=0.01 * 12 / 11,
        max_percentile=0.4 * 12 / 11,
    )
    torch.testing.assert_close(vocab(time).sigma, competitors(time).sigma)
    clipped = OneHotRankSchedule(12, inversion="analytic", sigma_min=1, sigma_max=2)
    result = clipped(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(result.sigma, torch.tensor([1.0, 2.0]))
    assert result.dsigma_dt.eq(0).all()
    with pytest.raises(ValueError, match="positive minimum"):
        OneHotRankSchedule(12, coupling="geometric", min_percentile=0)


def test_exponential_ve_values_derivatives_and_adaptive_composition():
    schedule = ExponentialVESchedule(0.01, 10)
    time = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
    path = schedule(time)
    torch.testing.assert_close(path.sigma, 0.01 * 1000**time)
    delta = 1e-6
    middle = time[1:3]
    derivative = (schedule(middle + delta).sigma - schedule(middle - delta).sigma) / (
        2 * delta
    )
    torch.testing.assert_close(schedule(middle).dsigma_dt, derivative)
    wrapped = CDCDGaussianSchedule(schedule)
    torch.testing.assert_close(wrapped(time).sigma, path.sigma)
    torch.testing.assert_close(schedule.training_time(time), time)
    for kwargs in ({"sigma_min": 0}, {"sigma_max": float("nan")}, {"sigma_min": 12}):
        with pytest.raises(ValueError):
            ExponentialVESchedule(**kwargs)


@pytest.mark.parametrize("representation", ["one_hot", "embedding"])
def test_exponential_ve_gaussian_training_and_sampling(representation):
    cfg = config(
        "gaussian_diffusion_sudoku",
        None,
        "noise_schedule=exponential_ve",
        f"representation={representation}",
    )
    algorithm = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 9, (2, 10))
    mask = torch.ones_like(tokens)
    mask[:, :3] = 0
    loss = algorithm.loss(tokens, mask)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    result, _ = algorithm.generate(instantiate(cfg.sampler), tokens, ~mask.bool())
    assert torch.equal(result[:, :3], tokens[:, :3])
