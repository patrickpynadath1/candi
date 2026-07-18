import pytest
import torch

from algos import algo_utils


@pytest.fixture
def schedule():
    return torch.tensor([0.1, 0.5, 1.0]), torch.tensor([0.2, 0.6, 0.8])


def test_sigma_from_time_interpolates(schedule):
    sigmas, errors = schedule
    result = algo_utils.sigma_from_time_vectorized(torch.tensor([0.4, 0.7]), sigmas, errors)
    assert torch.allclose(result, torch.tensor([0.3, 0.75]), atol=1e-6)


def test_sigma_from_time_clamps_to_table_boundaries(schedule):
    sigmas, errors = schedule
    result = algo_utils.sigma_from_time_vectorized(torch.tensor([-1.0, 0.0, 0.9, 2.0]), sigmas, errors)
    assert torch.allclose(result, torch.tensor([0.1, 0.1, 1.0, 1.0]))


def test_error_to_sigma_interpolation_and_boundary_clamping(schedule):
    sigmas, errors = schedule
    result = algo_utils.error_to_sigma(torch.tensor([0.0, 0.4, 1.0]), sigmas, errors)
    assert torch.allclose(result, torch.tensor([0.1, 0.3, 1.0]), atol=1e-6)


def test_sigma_to_error_interpolation_and_boundary_clamping(schedule):
    sigmas, errors = schedule
    result = algo_utils.sigma_to_error(torch.tensor([-1.0, 0.3, 2.0]), sigmas, errors)
    assert torch.allclose(result, torch.tensor([0.2, 0.4, 0.8]), atol=1e-6)


def test_training_sigma_schedule_endpoints_and_monotonicity():
    times = torch.linspace(0, 1, 11)
    values = algo_utils.training_sigma_ve(times, sigma_min=0.01, sigma_max=2.0)
    assert values[0].item() == pytest.approx(0.01)
    assert values[-1].item() == pytest.approx(2.0)
    assert torch.all(torch.diff(values) > 0)


def test_inference_sigma_schedule_endpoints_and_monotonicity():
    values = algo_utils.inference_sigmas(11, sigma_min=0.01, sigma_max=2.0)
    assert values.shape == (11,)
    assert values[0].item() == pytest.approx(2.0)
    assert values[-1].item() == pytest.approx(0.01)
    assert torch.all(torch.diff(values) < 0)


def test_sample_categorical_shape_and_valid_range():
    probabilities = torch.tensor([[[0.1, 0.2, 0.7]]]).expand(4, 5, 3)
    samples = algo_utils.sample_categorical(probabilities)
    assert samples.shape == (4, 5)
    assert samples.dtype == torch.long
    assert samples.min() >= 0
    assert samples.max() < 3


def test_sample_categorical_one_hot_is_deterministic():
    probabilities = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.equal(algo_utils.sample_categorical(probabilities), torch.tensor([1, 0]))


def test_expected_rank_is_bounded_and_increases_with_noise():
    ranks = algo_utils.expected_rank(10, torch.tensor(1.0), torch.tensor([0.01, 1.0, 100.0]))
    assert torch.all((ranks >= 0) & (ranks <= 0.9))
    assert torch.all(torch.diff(ranks) > 0)
