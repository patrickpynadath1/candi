import torch

from samplers.base import SamplingRequest
from samplers.runner import run_sampler
from samplers.semicat_dopri5 import SemicatDopri5Sampler

from .helpers import ModelBase


class DopriModel(ModelBase):
    sampling_family = "semicat"
    sampling_capabilities = frozenset({"continuous_prior", "semicat_vector_field"})

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, self.vocab_size)

    def sample_semicat_vector_field(self, _time, x):
        velocity = torch.zeros_like(x)
        velocity[..., 1] = 1
        return velocity


def test_dopri5_integrates_and_forwards_solver_options():
    result = run_sampler(
        SemicatDopri5Sampler(), DopriModel(),
        SamplingRequest(
            num_samples=1,
            options={"rtol": 1e-4, "atol": 1e-4},
        ),
    )

    assert result.samples.tolist() == [[1, 1, 1]]
    assert result.metadata["nfe"] is None
