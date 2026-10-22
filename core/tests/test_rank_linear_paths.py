"""Analytical identities, numerical inversion, and actual-corruption simulations."""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.special import betainc

from core.noise_schedules.simplex_rank import SimplexRankSchedule, simplex_expected_rank
from core.noise_schedules.spherical_rank import (
    SphericalRankSchedule,
    spherical_expected_rank,
)
from core.processes.dirichlet import DirichletProcess
from core.processes.spherical import SphericalProcess
from core.sweeps.configuration import compose
from core.training.module import TrainingModule


@pytest.mark.parametrize("vocab", [12, 260])
def test_simplex_rank_inversion_and_simulation(vocab):
    torch.manual_seed(19)
    schedule = SimplexRankSchedule(vocab)
    t = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    path = schedule(t)
    expected = 0.01 + (t.numpy() - 0.01) / 0.98 * 0.48
    analytic = simplex_expected_rank(
        path.time.numpy(), path.concentration.numpy(), vocab
    )
    np.testing.assert_allclose(analytic, expected, atol=3e-5)
    # Full actual Dirichlet sampler, not a Beta draw reused from the derivation.
    n = 30000
    tokens = torch.zeros(3, n, dtype=torch.long)
    z = DirichletProcess(vocab).corrupt(tokens, path)
    empirical = (z[..., 1] > z[..., 0]).double().mean(-1).numpy()
    np.testing.assert_allclose(empirical, expected, atol=0.012)
    print("simplex", vocab, "target", expected, "simulation", empirical)
    endpoints = schedule(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        endpoints.time, torch.tensor([0.0, 1.0], dtype=torch.float64)
    )


def test_simplex_binary_identity():
    u = np.array([0.1, 0.5, 0.9])
    c = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(simplex_expected_rank(u, c, 2), betainc(2 - u, u, 0.5))
    schedule = SimplexRankSchedule(3)
    with pytest.raises(ValueError, match="uniform reference"):
        schedule.validate_process(DirichletProcess(3, reference=[1, 2, 1]))


@pytest.mark.parametrize("dimension", [3, 32, 512])
def test_spherical_rank_inversion_and_actual_slerp_simulation(dimension):
    torch.manual_seed(23)
    schedule = SphericalRankSchedule(dimension)
    t = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    keep, _ = schedule(t)
    expected = 0.01 + (t.numpy() - 0.01) / 0.98 * 0.48
    np.testing.assert_allclose(
        spherical_expected_rank(1 - keep.numpy(), dimension, 512), expected, atol=3e-5
    )
    n = 30000
    # Fix target by rotational symmetry; independently draw noise and competitors.
    clean = torch.zeros(3, n, dimension, dtype=torch.float64)
    clean[..., 0] = 1
    process = SphericalProcess()
    z = process.corrupt(clean, keep, torch.ones(3, n, dtype=torch.bool))
    competitor = torch.nn.functional.normalize(torch.randn_like(z), dim=-1)
    empirical = ((z * competitor).sum(-1) > z[..., 0]).double().mean(-1).numpy()
    np.testing.assert_allclose(empirical, expected, atol=0.012)
    print("spherical", dimension, "target", expected, "simulation", empirical)


def test_circle_exact_rank_and_schedule_derivative():
    u = np.linspace(0, 1, 21)
    # In 2D theta is uniform on [0,pi], cap probability is u*theta/pi.
    np.testing.assert_allclose(spherical_expected_rank(u, 2), u / 2, atol=1e-12)
    schedule = SphericalRankSchedule(32)
    t = torch.tensor([0.005, 0.211, 0.611, 0.995], dtype=torch.float64)
    keep, derivative = schedule(t)
    h = 1e-7
    finite = (schedule(t + h)[0] - schedule(t - h)[0]) / (2 * h)
    torch.testing.assert_close(derivative, finite, rtol=1e-5, atol=1e-6)
    ends = schedule(torch.tensor([0.0, 1.0]))[0]
    torch.testing.assert_close(ends, torch.tensor([1.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("family", ["simplex", "spherical"])
@pytest.mark.parametrize("discrete", ["masked", "uniform"])
@pytest.mark.parametrize("sc", [False, True])
def test_rank_schedule_hybrid_train_and_generate(family, discrete, sc):
    config = compose(
        "corruption_factorial",
        {
            "hybrid_view@model.views.content": family + "_rank",
            "model.backbone.hidden_size": 16,
            "model.backbone.n_heads": 2,
            "model.backbone.n_blocks": 1,
            "model.backbone.cond_dim": 8,
            "training.ema": 0,
            "self_conditioning.enabled": sc,
            "sampler.steps": 3,
        },
    )
    config.model.views.discrete = OmegaConf.load(f"configs/hybrid_view/{discrete}.yaml")
    algorithm = TrainingModule(config).algorithm
    tokens = torch.randint(0, 12, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    loss = algorithm.loss(tokens, mask, training=True)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    samples, info = algorithm.generate(instantiate(config.sampler), tokens, ~mask)
    assert (samples >= 0).all() and (samples < 12).all()
    assert torch.equal(samples[:, 0], tokens[:, 0])
    assert info["nfe"] == 4
