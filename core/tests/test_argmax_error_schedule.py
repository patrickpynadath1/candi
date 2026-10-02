"""Calibration to argmax mistakes, independently checked against Gaussian integrals."""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from scipy.integrate import quad
from scipy.special import ndtr

from core.noise_schedules.argmax_error import OneHotArgmaxErrorSchedule
from core.noise_schedules.adaptive import CDCDGaussianSchedule
from core.representations.spec import RepresentationSpec
from core.tests.test_uniform_sedd_duo import config


def error_probability(vocab, sigma):
    correct = quad(
        lambda z: (
            np.exp(-z * z / 2) / np.sqrt(2 * np.pi) * ndtr(z + 1 / sigma) ** (vocab - 1)
        ),
        -12,
        12,
        epsabs=1e-11,
    )[0]
    return 1 - correct


@pytest.mark.parametrize("vocab", [2, 12, 50257])
def test_table_matches_independent_quadrature(vocab):
    schedule = OneHotArgmaxErrorSchedule(vocab, num_points=41)
    for index in (0, 9, 20, 40):
        sigma = schedule.sigmas[index].item()
        assert schedule.errors[index].item() == pytest.approx(
            error_probability(vocab, sigma), abs=2e-8
        )
    if vocab == 2:
        expected = torch.special.ndtr(-1 / (2**0.5 * schedule.sigmas))
        torch.testing.assert_close(schedule.errors, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("coupling", ["linear", "geometric"])
def test_error_targets_derivatives_and_resume(coupling):
    schedule = OneHotArgmaxErrorSchedule(12, coupling=coupling)
    time = torch.tensor([0.13137, 0.41234, 0.73337], dtype=torch.float64)
    path = schedule(time)
    target = (
        schedule.error_min + time * (schedule.error_max - schedule.error_min)
        if coupling == "linear"
        else schedule.error_min * (schedule.error_max / schedule.error_min) ** time
    )
    actual = torch.tensor(
        [error_probability(12, s.item()) for s in path.sigma], dtype=time.dtype
    )
    torch.testing.assert_close(actual, target, atol=3e-6, rtol=2e-5)
    epsilon = 1e-7
    difference = (schedule(time + epsilon).sigma - schedule(time - epsilon).sigma) / (
        2 * epsilon
    )
    torch.testing.assert_close(path.dsigma_dt, difference, atol=1e-7, rtol=1e-6)
    ends = schedule(torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(
        ends.sigma, torch.tensor([0.2, 4.0], dtype=torch.float64)
    )
    restored = OneHotArgmaxErrorSchedule(12, coupling=coupling)
    restored.load_state_dict(schedule.state_dict())
    torch.testing.assert_close(restored(time).sigma, path.sigma, rtol=0, atol=0)
    wrapped = CDCDGaussianSchedule(schedule)
    torch.testing.assert_close(wrapped(time).sigma, path.sigma)
    with pytest.raises(ValueError, match="representation kind"):
        schedule.representation_requirements.validate(
            RepresentationSpec("embedding", 12, 12, True), "argmax_error"
        )


def test_invalid_targets_reject_silent_miscalibration():
    with pytest.raises(ValueError, match="outside the sigma table"):
        OneHotArgmaxErrorSchedule(12, error_max=0.9)
    with pytest.raises(ValueError, match="random-guess"):
        OneHotArgmaxErrorSchedule(2, error_max=0.8)
    schedule = OneHotArgmaxErrorSchedule(12)
    with pytest.raises(ValueError, match="schedule time"):
        schedule(torch.tensor([float("nan")]))


def test_hydra_training_sampling_and_adaptive_base():
    cfg = config("gaussian_diffusion_sudoku", None, "noise_schedule=argmax_error")
    algorithm = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 9, (2, 10))
    mask = torch.ones_like(tokens)
    mask[:, :3] = 0
    loss = algorithm.loss(tokens, mask)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in algorithm.parameters()
        if p.grad is not None
    )
    samples, _ = algorithm.generate(instantiate(cfg.sampler), tokens, ~mask.bool())
    assert torch.equal(samples[:, :3], tokens[:, :3])
    adaptive = config(
        "gaussian_diffusion_sudoku",
        None,
        "noise_schedule=cdcd_gaussian_adaptive",
        "noise_schedule@noise_schedule.base=argmax_error",
    )
    instantiate(adaptive.algorithm)
