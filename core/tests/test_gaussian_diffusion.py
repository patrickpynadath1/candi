"""Gaussian schedule, process, predictor and Euler integration contracts."""

from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from core.noise_schedules.gaussian_rank import GaussianRankSchedule
from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.noise_schedules.utils import (
    expected_rank,
    training_sigma_ve,
    inference_sigmas,
    error_to_sigma,
    sigma_to_error,
    build_error_to_sigma_schedule,
)
from core.processes.gaussian import GaussianProcess
from core.representations.one_hot import OneHotRepresentation
from core.parameterizations.posterior import PosteriorParameterization
from core.samplers.euler import EulerSampler


def test_rank_lookup_and_buffers():
    schedule = GaussianRankSchedule(12)
    expected = 11 / 12 * 0.5 * (1 + torch.erf(-1 / (2 * schedule.sigmas)))
    torch.testing.assert_close(schedule.ranks, expected)
    assert set(dict(schedule.named_buffers())) == {"sigmas", "ranks"}
    values = schedule(torch.linspace(0, 1, 101)).sigma
    assert torch.all(values[1:] >= values[:-1])
    assert 0.2 <= values.min() < values.max() <= 4
    torch.testing.assert_close(
        schedule.training_time(torch.tensor([1.0])), torch.tensor([0.999])
    )
    for kwargs in ({"sigma_max": float("inf")}, {"sigma_min": -1}, {"num_points": 1}):
        with pytest.raises(ValueError):
            GaussianRankSchedule(12, **kwargs)
    with pytest.raises(ValueError):
        schedule(torch.tensor([float("nan")]))


def test_schedule_utilities_roundtrip():
    sigmas, errors = build_error_to_sigma_schedule(3, (0.2, 1.0), 20)
    torch.testing.assert_close(error_to_sigma(errors, sigmas, errors), sigmas)
    torch.testing.assert_close(sigma_to_error(sigmas, sigmas, errors), errors)
    torch.testing.assert_close(
        training_sigma_ve(torch.tensor([0.0, 1.0]), 0.2, 4), torch.tensor([0.2, 4.0])
    )
    torch.testing.assert_close(
        inference_sigmas(5, 0.2, 4)[[0, -1]], torch.tensor([4.0, 0.2])
    )
    assert expected_rank(3, torch.tensor(1.0), torch.tensor(0.5)).ndim == 0


def test_gaussian_corruption_and_vector_field():
    process = GaussianProcess()
    representation = OneHotRepresentation(3)
    tokens = torch.tensor([[0, 1]])
    torch.manual_seed(10)
    state = process.corrupt(
        representation.encode(tokens),
        torch.ones(1),
        torch.tensor([0.5]),
        torch.tensor([[0, 1]]),
    )
    assert torch.equal(state[:, 0], representation.encode(tokens)[:, 0])
    assert not torch.equal(state[:, 1], representation.encode(tokens)[:, 1])
    mean = representation.encode(tokens)
    torch.testing.assert_close(
        process.reverse_vector_field(
            state,
            mean,
            torch.ones(1),
            torch.tensor([0.5]),
            torch.zeros(1),
            torch.ones(1),
        ),
        2 * (state - mean),
    )


def test_unconstrained_posterior_training_does_not_shape():
    logits = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    state = torch.randn_like(logits)
    predictor = PosteriorParameterization(temperature=0.1, top_k=1)
    torch.testing.assert_close(
        predictor.training_posterior(logits, state), logits.log_softmax(-1)
    )
    assert predictor.sampling_posterior(logits, state).exp().tolist() == [
        [[0.0, 0.0, 1.0]]
    ]
    (-predictor.training_posterior(logits, state)[..., 0]).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_euler_preserves_prompts_and_matches_constant_mean_solution():
    process = GaussianProcess()
    representation = OneHotRepresentation(3)
    prompt = torch.tensor([[2, 0]])
    mask = torch.tensor([[True, False]])
    calls = []
    states = []

    def mean(state, time, path):
        calls.append((time.clone(), path.sigma.clone()))
        states.append(state.clone())
        assert torch.equal(state[:, 0], representation.encode(prompt)[:, 0])
        return representation.encode(torch.ones_like(prompt))

    algo = SimpleNamespace(
        predictor=PosteriorParameterization(),
        process=process,
        representation=representation,
        schedule=LinearInterpolationSchedule(),
        posterior_mean=mean,
        sampling_prediction=lambda state, time, path: mean(state, time, path).log(),
    )
    output, metadata = EulerSampler(steps=3).sample(algo, prompt, mask)
    assert output.tolist() == [[2, 1]]
    assert metadata == {"nfe": 4, "nfe_per_sample": [4]}
    # With a constant mean the linear path has constant time velocity.
    clean = representation.encode(torch.ones_like(prompt))
    expected = clean + calls[-1][1][:, None, None] / calls[0][1][:, None, None] * (
        states[0] - clean
    )
    torch.testing.assert_close(states[-1][:, 1], expected[:, 1])
    assert calls[0][0].item() == pytest.approx(0.999)


@pytest.mark.parametrize("schedule", ["gaussian_rank", "linear_interpolation"])
def test_gaussian_hydra_single_example_loss_and_sampling(schedule):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=gaussian_diffusion_sudoku",
                "model.hidden_size=32",
                "model.n_heads=4",
                "model.n_blocks=1",
                "model.cond_dim=16",
                f"noise_schedule={schedule}",
            ],
        )
    algo = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 12, (1, 10))
    mask = torch.tensor([[0] * 5 + [1] * 5])
    loss = algo.loss(tokens, mask)
    assert torch.isfinite(loss["loss"]) and loss["num_tokens"] == 5
    loss["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    samples, stats = algo.generate(EulerSampler(2), tokens, ~mask.bool())
    assert samples.shape == tokens.shape
    assert torch.equal(samples[:, :5], tokens[:, :5])
    assert stats["nfe"] == 3
    if schedule == "gaussian_rank":
        # General alpha=1 path keeps the original VE training objective exactly.
        algo.eval()
        torch.manual_seed(29)
        actual = algo.loss(tokens, mask)["loss"]
        torch.manual_seed(29)
        time = 0.999 * algo.time_sampler.sample(len(tokens), tokens.device)
        sigma = algo.schedule(time).sigma
        clean = algo.representation.encode(tokens)
        state = clean + sigma[:, None, None] * torch.randn_like(clean)
        state = torch.where(mask.bool()[..., None], state, clean)
        logits = algo.model(state, time, sigma, torch.ones_like(sigma))
        ce = (
            -(logits - logits.logsumexp(-1, keepdim=True))
            .gather(-1, tokens[..., None])
            .squeeze(-1)
        )
        expected = (ce * mask).sum() / mask.sum()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_linear_path_forward_endpoints_and_velocity():
    schedule = LinearInterpolationSchedule()
    path = schedule(torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(path.alpha, torch.tensor([1.0, 0.5, 0.0]))
    torch.testing.assert_close(path.sigma, torch.tensor([0.0, 0.5, 1.0]))
    process = GaussianProcess()
    representation = OneHotRepresentation(3)
    clean = representation.encode(torch.tensor([[1, 2]]))
    eps = torch.randn_like(clean)
    path = schedule(torch.tensor([0.3]))
    state = path.alpha[:, None, None] * clean + path.sigma[:, None, None] * eps
    field = process.reverse_vector_field(
        state, clean, path.alpha, path.sigma, path.dalpha_dt, path.dsigma_dt
    )
    torch.testing.assert_close(field, eps - clean)
    with pytest.raises(ValueError, match="positive sigma"):
        process.reverse_vector_field(
            state, clean, torch.ones(1), torch.zeros(1), -torch.ones(1), torch.ones(1)
        )
    tokens = torch.tensor([[0, 1]])
    mask = torch.tensor([[False, True]])
    torch.manual_seed(3)
    state = process.corrupt(representation.encode(tokens), path.alpha, path.sigma, mask)
    torch.manual_seed(3)
    expected = path.alpha[:, None, None] * representation.encode(tokens) + path.sigma[
        :, None, None
    ] * torch.randn_like(state)
    torch.testing.assert_close(state[:, 1], expected[:, 1])
    torch.testing.assert_close(state[:, 0], representation.encode(tokens)[:, 0])


def test_rank_derivative_and_constant_signal_corruption_parity():
    schedule = GaussianRankSchedule(12)
    # Choose the middle of lookup cells, away from derivative discontinuities.
    target = (schedule.ranks[[100, 400, 700]] + schedule.ranks[[101, 401, 701]]) / 2
    time = ((target - 0.01) / 0.39).requires_grad_()
    path = schedule(time)
    (derivative,) = torch.autograd.grad(path.sigma.sum(), time)
    torch.testing.assert_close(path.dsigma_dt, derivative)
    clamped = GaussianRankSchedule(12, min_percentile=0.0, max_percentile=1.0)(
        torch.tensor([0.0, 1.0])
    )
    assert torch.equal(clamped.dsigma_dt, torch.zeros(2))
    tokens = torch.tensor([[1, 2], [0, 1]])
    process = GaussianProcess()
    representation = OneHotRepresentation(3)
    sigma = torch.tensor([0.3, 0.5])
    torch.manual_seed(4)
    state = process.corrupt(
        representation.encode(tokens), torch.ones(2), sigma, torch.ones_like(tokens)
    )
    torch.manual_seed(4)
    torch.testing.assert_close(
        state,
        representation.encode(tokens) + torch.randn_like(state) * sigma[:, None, None],
        rtol=0,
        atol=0,
    )
