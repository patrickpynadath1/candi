import torch

from samplers.base import SamplingRequest
from samplers.runner import run_sampler
from samplers.semicat_flow_map import SemicatFlowMapSampler

from .helpers import ModelBase


class FlowMapModel(ModelBase):
    sampling_family = "semicat"
    sampling_capabilities = frozenset({"continuous_prior", "semicat_flow_map"})

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, self.vocab_size)

    def sample_semicat_flow_map(self, x, _start, _end):
        result = torch.zeros_like(x)
        result[..., 2] = 1
        return result


def test_flow_map_steps_and_projects_prompt_at_the_end():
    request = SamplingRequest(
        num_samples=1, num_steps=2,
        prompt_tokens=torch.tensor([[1, 0, 0]]),
        prompt_mask=torch.tensor([[1, 0, 0]]),
    )
    result = run_sampler(SemicatFlowMapSampler(), FlowMapModel(), request)

    assert result.samples.tolist() == [[1, 2, 2]]
    assert result.metadata["nfe"] == 2
