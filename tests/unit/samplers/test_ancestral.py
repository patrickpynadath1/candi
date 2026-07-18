import torch

from samplers.ancestral import AncestralSampler
from samplers.base import SamplingRequest
from samplers.runner import run_sampler

from .helpers import ModelBase


class AncestralModel(ModelBase):
    sampling_family = "absorbing_diffusion"
    sampling_capabilities = frozenset({
        "discrete_prior", "noise_schedule", "ancestral_transition",
        "greedy_denoise",
    })

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, dtype=torch.long)

    def noise(self, time):
        return None, 1 - time / 2

    def sample_ancestral_transition(self, *, x, prompt_mask=None, **_kwargs):
        updated = x + 1
        if prompt_mask is not None:
            updated = torch.where(prompt_mask.bool(), x, updated)
        return None, updated

    def sample_greedy_denoise(self, *, x, t):
        del t
        return x + 5


def test_runs_every_step_and_preserves_prompt_positions():
    request = SamplingRequest(
        num_samples=1,
        num_steps=2,
        prompt_tokens=torch.tensor([[3, 0, 0]]),
        prompt_mask=torch.tensor([[1, 0, 0]]),
    )
    result = run_sampler(AncestralSampler("none"), AncestralModel(), request)

    assert result.samples.tolist() == [[3, 2, 2]]
    assert result.metadata["nfe"] == 2


def test_greedy_noise_removal_is_an_explicit_extra_evaluation():
    result = run_sampler(
        AncestralSampler("greedy"),
        AncestralModel(),
        SamplingRequest(num_samples=1, num_steps=1),
    )

    assert torch.equal(result.samples, torch.full((1, 3), 6))
    assert result.metadata["nfe"] == 2
