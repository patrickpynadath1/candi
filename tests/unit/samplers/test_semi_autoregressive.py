import torch

from samplers.base import SamplingRequest
from samplers.runner import run_sampler
from samplers.semi_autoregressive import SemiAutoregressiveSampler

from .helpers import ModelBase


class SemiARModel(ModelBase):
    sampling_family = "absorbing_diffusion"
    sampling_capabilities = frozenset({
        "discrete_prior", "ancestral_transition", "zero_time_denoise",
    })
    time_conditioning = False

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, dtype=torch.long)

    def sample_ancestral_transition(self, *, x, **_kwargs):
        return None, x + 1

    def sample_zero_time_denoise(self, x, _ones):
        return x


def test_generates_overlapping_strides_and_returns_chunks():
    result = run_sampler(
        SemiAutoregressiveSampler(),
        SemiARModel(),
        SamplingRequest(
            num_samples=1,
            options={"stride_length": 1, "num_strides": 1, "dt": 0.5},
        ),
    )

    assert result.samples.shape == (1, 4)
    assert len(result.metadata["chunks"]) == 3
    assert result.metadata["nfe"] == 6
