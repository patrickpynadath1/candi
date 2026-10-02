"""Representation geometry, schedule compatibility and shared solver integration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from hydra.errors import InstantiationException

from core.representations.one_hot import OneHotRepresentation
from core.representations.embedding import EmbeddingRepresentation
from core.noise_schedules.embedding_rank import (
    EmbeddingSphereRankSchedule,
    embedding_table_rank,
)
from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.processes.gaussian import GaussianProcess
from core.parameterizations.posterior import PosteriorParameterization
from core.samplers.euler import EulerSampler
from core.samplers.heun import HeunSampler
from core.samplers.ode import ODESampler


def configuration(*overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=gaussian_diffusion_sudoku",
                "model.hidden_size=16",
                "model.n_heads=2",
                "model.n_blocks=1",
                "model.cond_dim=8",
                "model.dropout=0",
                *overrides,
            ],
        )


def test_representation_operations_and_trainable_geometry():
    onehot = OneHotRepresentation(3)
    embedding = EmbeddingRepresentation(3, dimension=2, trainable=True)
    with torch.no_grad():
        embedding.table.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]]))
    tokens = torch.tensor([[2, 0, 1]])
    probabilities = torch.tensor([[[0.2, 0.3, 0.5]]])
    assert onehot.encode(tokens).shape == (1, 3, 3)
    torch.testing.assert_close(
        embedding.encode(tokens), torch.tensor([[[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    )
    torch.testing.assert_close(
        embedding.posterior_mean(probabilities), torch.tensor([[[-0.3, 0.3]]])
    )
    assert torch.equal(onehot.decode(onehot.encode(tokens)), tokens)
    assert torch.equal(embedding.decode(embedding.encode(tokens)), tokens)
    embedding.posterior_mean(probabilities).sum().backward()
    assert (
        embedding.table.grad is not None and torch.isfinite(embedding.table.grad).all()
    )
    torch.testing.assert_close(embedding.codebook().norm(dim=-1), torch.ones(3))


@pytest.mark.parametrize(
    "overrides,message",
    [
        (["representation=embedding"], "representation kind"),
        (["noise_schedule=embedding_sphere_rank"], "representation kind"),
        (
            [
                "representation=embedding",
                "noise_schedule=embedding_sphere_rank",
                "noise_schedule.dimension=8",
            ],
            "continuous dimension",
        ),
        (
            [
                "representation=embedding",
                "noise_schedule=embedding_sphere_rank",
                "representation.unit_normalized=false",
            ],
            "unit-normalized",
        ),
        (["noise_schedule.vocab_size=13"], "vocabulary size"),
        (["model.state_dim=17"], "state_dim"),
    ],
)
def test_incompatible_combinations_fail_at_construction(overrides, message):
    with pytest.raises(InstantiationException, match=message):
        instantiate(configuration(*overrides).algorithm)


@pytest.mark.parametrize(
    "representation,schedule",
    [
        ("one_hot", "one_hot_rank"),
        ("one_hot", "linear_interpolation"),
        ("embedding", "linear_interpolation"),
        ("embedding", "embedding_sphere_rank"),
    ],
)
@pytest.mark.parametrize("sampler_class", [EulerSampler, HeunSampler, ODESampler])
def test_real_model_loss_and_all_samplers(representation, schedule, sampler_class):
    options = [f"representation={representation}", f"noise_schedule={schedule}"]
    if representation == "embedding":
        options += ["representation.dimension=7", "representation.trainable=true"]
    algo = instantiate(configuration(*options).algorithm)
    tokens = torch.randint(0, 12, (2, 6))
    fixed = torch.zeros_like(tokens, dtype=torch.bool)
    fixed[:, :3] = True
    loss = algo.loss(tokens, ~fixed)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    if representation == "embedding":
        assert algo.representation.table.grad is not None
    kwargs = (
        {"steps": 3}
        if sampler_class is not ODESampler
        else {"rtol": 0.05, "atol": 0.005}
    )
    for decoding in ("posterior", "representation"):
        samples, metadata = algo.generate(
            sampler_class(decoding=decoding, **kwargs), tokens, fixed
        )
        assert samples.shape == tokens.shape
        assert torch.equal(samples[fixed], tokens[fixed])
        assert ((samples >= 0) & (samples < 12)).all()
        assert metadata["nfe"] > 1
    # Representation parameters and calibration tables survive checkpoint loading.
    restored = instantiate(configuration(*options).algorithm)
    restored.load_state_dict(algo.state_dict(), strict=True)
    torch.testing.assert_close(
        restored.representation.encode(tokens), algo.representation.encode(tokens)
    )


@pytest.mark.parametrize("coupling", ["linear", "geometric"])
@pytest.mark.parametrize("width", [0.0, 0.01])
def test_embedding_rank_derivatives_and_endpoints(coupling, width):
    schedule = EmbeddingSphereRankSchedule(7, coupling=coupling, endpoint_width=width)
    time = torch.tensor([0.0021, 0.1327, 0.5483, 0.8731, 0.9973], requires_grad=True)
    path = schedule(time)
    (derivative,) = torch.autograd.grad(path.sigma.sum(), time)
    torch.testing.assert_close(path.dsigma_dt, derivative)
    torch.testing.assert_close(path.dalpha_dt, -derivative)
    torch.testing.assert_close(path.alpha + path.sigma, torch.ones_like(time))
    assert torch.all(path.sigma[1:] >= path.sigma[:-1])
    endpoints = schedule(torch.tensor([0.0, 1.0])).sigma
    if width:
        torch.testing.assert_close(endpoints, torch.tensor([0.0, 1.0]))
    else:
        assert 0 < endpoints[0] < endpoints[1] < 1
    with pytest.raises(ValueError):
        schedule(torch.tensor([float("nan")]))


def test_embedding_rank_agrees_with_random_table_approximation():
    torch.manual_seed(19)
    table = torch.nn.functional.normalize(torch.randn(256, 16), dim=-1)
    schedule = EmbeddingSphereRankSchedule(16, coupling="linear", endpoint_width=0)
    times = torch.tensor([0.15, 0.5, 0.85])
    path = schedule(times)
    actual = embedding_table_rank(table, path.sigma)
    expected = schedule.rank_min + times * (schedule.rank_max - schedule.rank_min)
    torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0)


def test_representation_decoding_is_distinct_from_posterior_decoding():
    representation = EmbeddingRepresentation(3, dimension=2)
    representation.table.copy_(torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]]))
    p = torch.tensor([[[0.4, 0.35, 0.25]]])
    algo = SimpleNamespace(
        predictor=PosteriorParameterization(),
        representation=representation,
        process=GaussianProcess(),
        schedule=LinearInterpolationSchedule(),
        sampling_prediction=lambda state, time, path: p.log(),
        posterior_mean=lambda state, time, path: representation.posterior_mean(p),
    )
    tokens, mask = (
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.bool),
    )
    assert (
        EulerSampler(2, decoding="posterior").sample(algo, tokens, mask)[0].item() == 0
    )
    assert (
        EulerSampler(2, decoding="representation").sample(algo, tokens, mask)[0].item()
        == 1
    )
