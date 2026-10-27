"""Schedule-independent Gaussian DDPM bridges and sampler integration."""

from types import SimpleNamespace

import pytest
import torch

from core.noise_schedules.adaptive.gaussian import CDCDGaussianSchedule
from core.noise_schedules.gaussian_path import GaussianPath
from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.noise_schedules.exponential_ve import ExponentialVESchedule
from core.noise_schedules.rescaled_gaussian import RescaledGaussianSchedule
from core.parameterizations.posterior import PosteriorParameterization
from core.processes.gaussian import GaussianProcess
from core.representations.embedding import EmbeddingRepresentation
from core.representations.one_hot import OneHotRepresentation
from core.samplers.gaussian_ddpm import GaussianDDPMSampler
from core.samplers.hybrid_updates import GaussianPosteriorUpdate


def path(alpha, sigma):
    alpha, sigma = (
        torch.tensor([alpha], dtype=torch.float64),
        torch.tensor([sigma], dtype=torch.float64),
    )
    return GaussianPath(alpha, sigma, torch.zeros_like(alpha), torch.zeros_like(sigma))


@pytest.mark.parametrize(
    "source,destination",
    [
        ((0.0, 1.0), (0.4, 0.6)),  # exact pure-noise start, formerly 0 * infinity
        ((1e-30, 1.0), (0.4, 0.6)),
        ((0.4, 0.6), (1.0, 0.0)),  # exact clean endpoint
        ((0.0, 2.0), (0.0, 1.0)),  # equal zero SNR, deterministic rescale
        ((1.0, 0.0), (2.0, 0.0)),  # equal infinite SNR
        ((0.4, 0.6), (0.4, 0.6)),
    ],
)
def test_endpoints(source, destination):
    state = torch.randn(1, 3, 4, dtype=torch.float64)
    clean, noise = torch.randn_like(state), torch.randn_like(state)
    result = GaussianProcess().reverse_posterior(
        state, clean, path(*source), path(*destination), noise
    )
    assert torch.isfinite(result).all()
    if destination[1] == 0 and source[1] != 0:
        torch.testing.assert_close(result, clean * destination[0])
    if source[0] == 0 and destination[0] > 0:
        torch.testing.assert_close(
            result, clean * destination[0] + noise * destination[1]
        )
    if source == destination:
        torch.testing.assert_close(result, state)


@pytest.mark.parametrize(
    "source,destination",
    [((0.5, 0.5), (0.4, 0.6)), ((1.0, 0.0), (1.0, 0.1)), ((1.0, 1.0), (0.0, 1.0))],
)
def test_rejects_decreasing_reverse_snr(source, destination):
    state = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="nondecreasing"):
        GaussianProcess().reverse_posterior(
            state, state, path(*source), path(*destination), state
        )


@pytest.mark.parametrize(
    "source,destination",
    [((1.0, 2.0), (1.0, 0.5)), ((0.6, 0.8), (0.8, 0.6)), ((0.0, 1.0), (0.6, 0.4))],
)
def test_conditional_mean_and_variance(source, destination):
    torch.manual_seed(29)
    state = torch.full((1, 100000, 1), 0.3, dtype=torch.float64)
    clean = torch.full_like(state, 0.7)
    a_t, s_t = source
    a_s, s_s = destination
    forward_gain = a_t / a_s
    covariance = forward_gain * s_s**2
    mean = a_s * 0.7 + covariance / s_t**2 * (0.3 - a_t * 0.7)
    variance = s_s**2 - covariance**2 / s_t**2
    result = GaussianProcess().reverse_posterior(
        state, clean, path(*source), path(*destination), torch.randn_like(state)
    )
    assert result.mean().item() == pytest.approx(mean, abs=0.005)
    assert result.var().item() == pytest.approx(variance, rel=0.015)


def test_state_rescaling_equivariance():
    torch.manual_seed(31)
    state, clean, noise = [torch.randn(1, 5, 4, dtype=torch.float64) for _ in range(3)]
    process = GaussianProcess()
    reference = process.reverse_posterior(
        state, clean, path(0.3, 0.8), path(0.7, 0.4), noise
    )
    scaled = process.reverse_posterior(
        state * 3, clean, path(0.9, 2.4), path(1.4, 0.8), noise
    )
    torch.testing.assert_close(scaled, reference * 2)


@pytest.mark.parametrize(
    "representation", [OneHotRepresentation(3), EmbeddingRepresentation(3, 5)]
)
@pytest.mark.parametrize(
    "schedule",
    [
        LinearInterpolationSchedule(),
        ExponentialVESchedule(),
        RescaledGaussianSchedule(ExponentialVESchedule()),
        CDCDGaussianSchedule(LinearInterpolationSchedule()),
    ],
)
def test_standalone_and_hybrid_update(representation, schedule):
    prompt = torch.tensor([[2, 0]])
    mask = torch.tensor([[True, False]])
    probabilities = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    seen = []

    def predict(state, time, coefficients):
        assert torch.isfinite(state).all()
        torch.testing.assert_close(state[:, 0], representation.encode(prompt)[:, 0])
        seen.append(state.clone())
        return probabilities.log()

    algorithm = SimpleNamespace(
        predictor=PosteriorParameterization(),
        representation=representation,
        process=GaussianProcess(),
        schedule=schedule,
        sampling_prediction=predict,
    )
    output, metadata = GaussianDDPMSampler(
        steps=3, epsilon=0, prior_mode="schedule", prompt_mode="clean"
    ).sample(algorithm, prompt, mask)
    assert output.tolist() == [[2, 1]]
    assert metadata["nfe"] == 4
    source = SimpleNamespace(coefficients=schedule(torch.ones(1)))
    destination = SimpleNamespace(coefficients=schedule(torch.full((1,), 0.5)))
    noise = torch.randn_like(seen[0])
    hybrid = GaussianPosteriorUpdate()(
        algorithm.process,
        representation,
        seen[0],
        probabilities,
        source,
        destination,
        noise=noise,
    )
    expected = algorithm.process.reverse_posterior(
        seen[0],
        representation.posterior_mean(probabilities),
        source.coefficients,
        destination.coefficients,
        noise,
    )
    torch.testing.assert_close(hybrid, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_zero_alpha_is_finite(dtype):
    state = torch.randn(1, 3, 4).to(dtype)
    result = GaussianProcess().reverse_posterior(
        state, state, path(0.0, 1.0), path(0.5, 0.5), torch.zeros_like(state)
    )
    assert result.dtype == dtype
    torch.testing.assert_close(result, state * 0.5)


def test_generic_config_and_reference_replaid_policies():
    from pathlib import Path
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        generic = compose(
            config_name="config",
            overrides=["+experiment=onehot_gaussian", "sampler=gaussian_ddpm"],
        )
        plaid = compose(config_name="config", overrides=["+experiment=replaid"])
        sudoku = compose(
            config_name="config", overrides=["+experiment=replaid_sudoku_hard"]
        )
    assert generic.sampler.prior_mode == "schedule"
    assert generic.sampler.prompt_mode == "clean"
    assert not generic.sampler.final_transition_noise
    assert plaid.sampler.prior_mode == "standard_normal"
    assert plaid.sampler.prompt_mode == "noisy"
    assert plaid.sampler.final_transition_noise
    assert plaid.sampler.self_conditioning_mode == "bootstrap"
    assert sudoku.sampler.prompt_mode == "clean"


@pytest.mark.parametrize("mode,calls", [("previous", 1), ("bootstrap", 2)])
def test_ddpm_budget_reserves_terminal_and_bootstrap_calls(mode, calls):
    sampler = GaussianDDPMSampler(self_conditioning_mode=mode)
    for budget in (4, 16, 128):
        steps = sampler.steps_for_nfe(budget)
        assert calls * (steps + 1) <= budget
        assert calls * (steps + 2) > budget
    with pytest.raises(ValueError):
        sampler.steps_for_nfe(1)
