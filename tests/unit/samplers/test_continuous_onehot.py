import torch

from samplers.base import SamplingRequest
from samplers.continuous_onehot import ContinuousOnehotSampler
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


class OnehotModel(ModelBase):
    sampling_family = "continuous_onehot"
    sampling_capabilities = frozenset({
        "continuous_prior", "continuous_schedule", "continuous_transition",
    })

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sampling_continuous_schedule(self, times):
        return times

    def sample_continuous_transition(self, *, x, clean_mask, **_kwargs):
        prediction = torch.zeros_like(x)
        prediction[..., 2] = 1
        prediction[clean_mask] = x[clean_mask]
        return prediction, prediction


def test_returns_argmax_tokens_and_preserves_prompt():
    request = SamplingRequest(
        num_samples=1,
        num_steps=2,
        prompt_tokens=torch.tensor([[1, 0, 0]]),
        prompt_mask=torch.tensor([[1, 0, 0]]),
    )
    result = run_sampler(ContinuousOnehotSampler(), OnehotModel(), request)

    assert result.samples.tolist() == [[1, 2, 2]]
    assert result.metadata["nfe"] == 2
