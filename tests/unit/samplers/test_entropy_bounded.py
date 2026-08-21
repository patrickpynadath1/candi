from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.entropy_bounded import EntropyBoundedSampler
from samplers.runner import run_sampler

from .helpers import ModelBase


class EntropyModel(ModelBase):
    sampling_family = "absorbing_diffusion"
    sampling_capabilities = frozenset({
        "discrete_prior", "denoising_log_probs",
    })
    mask_index = 3
    num_tokens = 3

    def __init__(self):
        self.calls = 0
        self.config = SimpleNamespace(
            sampling=SimpleNamespace(steps=3, entropy_bound=0.0)
        )

    def prior_sample(self, batch, length):
        return torch.full((batch, length), self.mask_index, dtype=torch.long)

    def sample_denoising_log_probs(self, x, t):
        del t
        self.calls += 1
        # Increasing uncertainty from left to right.  The mask token has zero
        # probability and accepted positions are deterministic copies.
        distributions = torch.tensor([
            [0.99, 0.005, 0.005, 0.0],
            [0.80, 0.10, 0.10, 0.0],
            [0.55, 0.225, 0.225, 0.0],
        ]).expand(x.shape[0], -1, -1)
        deterministic = torch.nn.functional.one_hot(x, 4).float()
        probs = torch.where((x == self.mask_index).unsqueeze(-1), distributions, deterministic)
        return probs.clamp_min(torch.finfo(probs.dtype).tiny).log()


def test_zero_bound_unmasks_one_position_per_evaluation():
    model = EntropyModel()
    result = run_sampler(
        EntropyBoundedSampler(0.0), model,
        SamplingRequest(num_samples=1),
    )

    assert result.metadata["nfe"] == 3
    assert result.metadata["accepted_per_step"] == [[1], [1], [1]]
    assert not (result.samples == model.mask_index).any()


def test_large_bound_unmasks_all_positions_in_parallel():
    model = EntropyModel()
    result = run_sampler(
        EntropyBoundedSampler(10.0), model,
        SamplingRequest(num_samples=1),
    )

    assert result.metadata["nfe"] == 1
    assert result.metadata["accepted_per_step"] == [[3]]


def test_greedy_entropy_bounded_commits_argmax_tokens():
    model = EntropyModel()
    result = run_sampler(
        EntropyBoundedSampler(10.0, greedy_clean_estimate=True), model,
        SamplingRequest(num_samples=2),
    )
    assert torch.equal(result.samples, torch.zeros_like(result.samples))
    assert result.metadata["greedy_clean_estimate"] is True


def test_prompt_positions_are_preserved_and_excluded_from_bound():
    model = EntropyModel()
    result = run_sampler(
        EntropyBoundedSampler(0.0), model,
        SamplingRequest(
            num_samples=1,
            prompt_tokens=torch.tensor([[2, 0, 0]]),
            prompt_mask=torch.tensor([[True, False, False]]),
        ),
    )

    assert result.samples[0, 0].item() == 2
    assert result.metadata["nfe"] == 2


def test_entropy_prefix_uses_sum_minus_max_bound():
    entropy = torch.tensor([0.1, 0.2, 0.7])
    assert EntropyBoundedSampler._prefix_size(entropy, 0.09) == 1
    assert EntropyBoundedSampler._prefix_size(entropy, 0.10) == 2
    assert EntropyBoundedSampler._prefix_size(entropy, 0.30) == 3
