from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.runner import run_sampler
from samplers.uniform_entropy_bounded import (
    UniformEntropyBoundedRenoiseSampler,
    UniformEntropyBoundedSampler,
)

from .helpers import ModelBase


class UniformEntropyModel(ModelBase):
    sampling_family = "uniform_diffusion"
    sampling_capabilities = frozenset({
        "discrete_prior", "noise_schedule", "uniform_entropy_transition",
    })
    num_tokens = 3

    def __init__(self):
        self.inputs = []
        self.config = SimpleNamespace(sampling=SimpleNamespace(steps=3))

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, dtype=torch.long)

    def noise(self, time):
        return None, time

    def sample_uniform_entropy_transition(
        self, x, t, dt, frozen_mask=None, greedy_clean_estimate=False
    ):
        del t, dt
        self.greedy_clean_estimate = greedy_clean_estimate
        self.inputs.append(x.clone())
        probs = torch.tensor([
            [0.99, 0.005, 0.005, 0.0],
            [0.80, 0.10, 0.10, 0.0],
            [0.55, 0.225, 0.225, 0.0],
        ]).expand(x.shape[0], -1, -1)
        continued = (x + 1) % 3
        if frozen_mask is not None:
            continued = torch.where(frozen_mask, x, continued)
        return probs.clamp_min(torch.finfo(probs.dtype).tiny).log(), continued


def test_rejected_uniform_sites_continue_reverse_trajectory():
    model = UniformEntropyModel()
    result = run_sampler(
        UniformEntropyBoundedSampler(0.0), model,
        SamplingRequest(num_samples=1, num_steps=3),
    )

    assert result.metadata["accepted_per_step"] == [[1], [1], [1]]
    # Position zero was frozen after pass one. Rejected positions advanced.
    assert model.inputs[1][0, 0] == result.samples[0, 0]
    assert torch.equal(model.inputs[1][0, 1:], torch.ones(2, dtype=torch.long))


def test_large_bound_accepts_uniform_sites_in_one_pass():
    result = run_sampler(
        UniformEntropyBoundedSampler(10.0), UniformEntropyModel(),
        SamplingRequest(num_samples=2, num_steps=3),
    )
    assert result.metadata["nfe"] == 1
    assert result.metadata["nfe_per_sample"] == [1, 1]


def test_greedy_uniform_eb_accepts_argmax_clean_tokens():
    model = UniformEntropyModel()
    result = run_sampler(
        UniformEntropyBoundedSampler(10.0, greedy_clean_estimate=True),
        model,
        SamplingRequest(num_samples=2, num_steps=3),
    )

    assert torch.equal(result.samples, torch.zeros_like(result.samples))
    assert result.metadata["greedy_clean_estimate"] is True
    assert model.greedy_clean_estimate is True


def test_renoise_variant_commits_passes_and_resets_every_rejection():
    model = UniformEntropyModel()
    result = run_sampler(
        UniformEntropyBoundedRenoiseSampler(0.0), model,
        SamplingRequest(num_samples=1, num_steps=3),
    )

    assert result.metadata["accepted_per_step"] == [[1], [1], [1]]
    assert result.metadata["rejected_transition"] == "fresh_uniform_prior"
    # The first accepted token remains as context. Unlike the posterior-path
    # sampler, both rejected sites are fresh prior samples (zero here).
    assert model.inputs[1][0, 0] == result.samples[0, 0]
    assert torch.equal(model.inputs[1][0, 1:], torch.zeros(2, dtype=torch.long))
    # No accepted mask was passed into the model's reverse transition.
    assert len(model.inputs) == 3
