import torch

from samplers.base import SamplingRequest
from samplers.candi import CANDISampler
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


class CANDIModel(ModelBase):
    sampling_family = "candi"
    sampling_capabilities = frozenset({
        "continuous_prior", "continuous_schedule",
        "candi_continuous_transition", "candi_discrete_transition",
    })

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sampling_continuous_schedule(self, times, steps):
        assert len(times) == steps + 1
        return times

    def sample_candi_continuous_transition(self, x, *_args, **_kwargs):
        prediction = torch.zeros_like(x)
        prediction[..., 2] = 1
        return prediction, prediction

    def sample_candi_discrete_transition(
        self, continuous, _clean_probs, _time, _dt, *, prev_clean_mask
    ):
        return continuous, prev_clean_mask


def test_uncached_sampler_alternates_continuous_and_discrete_updates():
    result = run_sampler(
        CANDISampler(), CANDIModel(), SamplingRequest(num_samples=2, num_steps=2)
    )

    assert torch.equal(result.samples, torch.full((2, 3), 2))
    assert result.metadata["nfe"] == 2
