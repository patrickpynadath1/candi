import torch

from samplers.analytic import AnalyticSampler
from samplers.base import SamplingRequest
from samplers.runner import run_sampler

from .helpers import ModelBase


class AnalyticModel(ModelBase):
    sampling_family = "absorbing_diffusion"
    sampling_capabilities = frozenset({
        "discrete_prior", "noise_schedule", "analytic_transition",
        "denoiser_transition",
    })

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, dtype=torch.long)

    def noise(self, time):
        return None, time

    def sample_analytic_transition(self, *, x, **_kwargs):
        return x + 1

    def sample_denoiser_transition(self, *, x, **_kwargs):
        return x + 10


def test_uses_analytic_transition_for_each_step():
    result = run_sampler(
        AnalyticSampler("none"),
        AnalyticModel(),
        SamplingRequest(num_samples=1, num_steps=3),
    )

    assert torch.equal(result.samples, torch.full((1, 3), 3))
    assert result.metadata["nfe"] == 3


def test_ancestral_noise_removal_uses_denoiser_capability():
    result = run_sampler(
        AnalyticSampler("ancestral"),
        AnalyticModel(),
        SamplingRequest(num_samples=1, num_steps=1),
    )

    assert torch.equal(result.samples, torch.full((1, 3), 11))
    assert result.metadata["nfe"] == 2
