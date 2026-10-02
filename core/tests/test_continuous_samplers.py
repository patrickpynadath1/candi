"""Numerical accuracy, conditioning and model-call accounting for ODE sampling."""

from types import SimpleNamespace
import pytest
import torch

from core.processes.gaussian import GaussianProcess
from core.representations.one_hot import OneHotRepresentation
from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.parameterizations.posterior import PosteriorParameterization
from core.samplers.euler import EulerSampler
from core.samplers.heun import HeunSampler
from core.samplers.ode import ODESampler


@pytest.mark.parametrize(
    "sampler", [EulerSampler(100), HeunSampler(20), ODESampler(rtol=1e-6, atol=1e-7)]
)
def test_integrates_known_exponential(sampler):
    initial = torch.tensor([1.0])
    result = sampler.integrate(lambda time, state: state, initial)
    expected = torch.exp(torch.tensor(sampler.epsilon - sampler.start_time))
    tolerance = 0.003 if isinstance(sampler, EulerSampler) else 0.0002
    torch.testing.assert_close(result[0], expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize(
    "sampler_class,expected_nfe",
    [(EulerSampler, 4), (HeunSampler, 7), (ODESampler, None)],
)
def test_posterior_adapter_conditioning_and_nfe(sampler_class, expected_nfe):
    process = GaussianProcess()
    representation = OneHotRepresentation(3)
    prompt = torch.tensor([[2, 0], [0, 1]])
    mask = torch.tensor([[True, False], [True, False]])
    calls = []

    def posterior(state, time, path):
        assert torch.equal(state[:, 0], representation.encode(prompt)[:, 0])
        assert torch.all((time >= 0) & (time <= 1))
        calls.append(time.clone())
        return representation.encode(torch.ones_like(prompt))

    algo = SimpleNamespace(
        predictor=PosteriorParameterization(),
        process=process,
        representation=representation,
        schedule=LinearInterpolationSchedule(),
        posterior_mean=posterior,
        sampling_prediction=lambda state, time, path: posterior(
            state, time, path
        ).log(),
    )
    sampler = sampler_class(**({"steps": 3} if expected_nfe else {}))
    result, stats = sampler.sample(algo, prompt, mask)
    assert result.tolist() == [[2, 1], [0, 1]]
    assert stats["nfe"] == len(calls)
    assert stats["nfe_per_sample"] == [len(calls)] * 2
    if expected_nfe:
        assert len(calls) == expected_nfe


def test_adaptive_budget_counts_calls_and_rejects_invalid_parameters():
    process = GaussianProcess()
    representation = OneHotRepresentation(3)
    calls = []

    def posterior(state, time, path):
        calls.append(time)
        return torch.zeros_like(state)

    algo = SimpleNamespace(
        predictor=PosteriorParameterization(),
        process=process,
        representation=representation,
        schedule=LinearInterpolationSchedule(),
        posterior_mean=posterior,
        sampling_prediction=lambda state, time, path: posterior(
            state, time, path
        ).log(),
    )
    with pytest.raises(RuntimeError, match="max_nfe=3"):
        ODESampler(max_nfe=3).sample(
            algo,
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.bool),
        )
    assert len(calls) == 2  # One evaluation reserved for final decoding.
    for kwargs in (
        {"rtol": float("nan")},
        {"atol": 0},
        {"max_steps": 0},
        {"max_nfe": 1},
    ):
        with pytest.raises(ValueError):
            ODESampler(**kwargs)
    for kwargs in (
        {"epsilon": float("nan")},
        {"start_time": float("inf")},
        {"steps": 1.5},
    ):
        with pytest.raises(ValueError):
            EulerSampler(**kwargs)


def test_requires_posterior_parameterization():
    with pytest.raises(TypeError, match="PosteriorParameterization"):
        EulerSampler().sample(
            SimpleNamespace(predictor=object()),
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.bool),
        )


def test_euler_matches_original_loop_exactly():
    from core.noise_schedules.gaussian_rank import GaussianRankSchedule

    process = GaussianProcess()
    representation = OneHotRepresentation(3)
    prompt = torch.tensor([[2, 0, 1], [0, 1, 2]])
    mask = torch.tensor([[True, False, False], [False, True, False]])
    states = []

    def posterior(state, time, path):
        states.append(state.clone())
        return state.softmax(-1)

    algo = SimpleNamespace(
        predictor=PosteriorParameterization(),
        process=process,
        representation=representation,
        schedule=GaussianRankSchedule(3),
        posterior_mean=posterior,
        sampling_prediction=lambda state, time, path: posterior(
            state, time, path
        ).log(),
    )
    sampler = EulerSampler(steps=9)
    torch.manual_seed(18)
    actual, stats = sampler.sample(algo, prompt, mask)
    actual_states = states[:]
    states.clear()
    torch.manual_seed(18)
    times = torch.linspace(sampler.start_time, sampler.epsilon, sampler.steps + 1)
    clean = representation.encode(prompt)
    state = torch.where(
        mask[..., None],
        clean,
        process.prior(clean, algo.schedule(times[0].expand(len(prompt))).sigma),
    )
    for time, next_time in zip(times[:-1], times[1:]):
        time_batch = time.expand(len(state))
        path = algo.schedule(time_batch)
        mean = posterior(state, time_batch, path)
        field = process.reverse_vector_field(
            state, mean, path.alpha, path.sigma, path.dalpha_dt, path.dsigma_dt
        )
        state = torch.where(mask[..., None], clean, state + (next_time - time) * field)
    time = times[-1].expand(len(state))
    expected = torch.where(
        mask, prompt, posterior(state, time, algo.schedule(time)).argmax(-1)
    )
    assert torch.equal(actual, expected)
    assert stats["nfe"] == 10
    for actual_state, expected_state in zip(actual_states, states):
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


def test_adaptive_step_budget_and_nonfinite_prediction():
    with pytest.raises(RuntimeError, match="adaptive ODE integration failed"):
        ODESampler(max_steps=1, rtol=1e-7, atol=1e-8).integrate(
            lambda time, state: state, torch.ones(1)
        )
    algo = SimpleNamespace(
        predictor=PosteriorParameterization(),
        process=GaussianProcess(),
        representation=OneHotRepresentation(3),
        schedule=LinearInterpolationSchedule(),
        posterior_mean=lambda state, time, path: torch.full_like(state, float("nan")),
    )
    with pytest.raises(ValueError, match="finite"):
        EulerSampler().sample(
            algo,
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.bool),
        )
