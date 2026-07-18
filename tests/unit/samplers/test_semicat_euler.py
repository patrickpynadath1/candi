import torch

from samplers.base import SamplingRequest
from samplers.runner import run_sampler
from samplers.semicat_euler import SemicatEulerSampler

from .helpers import ModelBase


class EulerModel(ModelBase):
    sampling_family = "semicat"
    sampling_capabilities = frozenset({"continuous_prior", "semicat_vector_field"})

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, self.vocab_size)

    def sample_semicat_vector_field(self, _time, x):
        velocity = torch.zeros_like(x)
        velocity[..., 3] = 1
        return velocity


def test_euler_integrates_vector_field_on_requested_grid():
    result = run_sampler(
        SemicatEulerSampler(), EulerModel(),
        SamplingRequest(num_samples=1, num_steps=4),
    )

    assert result.samples.tolist() == [[3, 3, 3]]
    assert result.metadata["nfe"] == 4
