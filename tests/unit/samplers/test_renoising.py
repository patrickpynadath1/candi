from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.renoising import RenoisingSampler
from samplers.runner import run_sampler

from .helpers import ModelBase


class RenoisingModel(ModelBase):
    sampling_family = "candi"
    sampling_capabilities = frozenset({
        "continuous_prior",
        "continuous_schedule",
        "candi_discrete_schedule",
        "candi_clean_distribution",
        "candi_forward_corruption",
    })
    num_tokens = 3

    def __init__(self):
        self.config = SimpleNamespace(sampling=SimpleNamespace(steps=3))
        self.forward_noise = []

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, 2)

    def sampling_prompt_state(self, state, prompt_tokens, prompt_mask):
        reveal = torch.zeros(state.shape[:2], dtype=torch.bool)
        if prompt_tokens is None:
            return state, reveal
        reveal = prompt_mask.bool()
        return state, reveal

    def sampling_continuous_schedule(self, times, steps):
        del steps
        return times

    def sampling_discrete_schedule(self, times):
        return times

    def sample_candi_clean_distribution(self, state, *_args):
        probs = torch.zeros(*state.shape[:2], 5)
        probs[..., 2] = 1
        return probs

    def sample_candi_forward_corruption(
        self, clean_tokens, discrete_noise, continuous_noise, valid_tokens
    ):
        self.forward_noise.append((float(discrete_noise), float(continuous_noise)))
        state = torch.stack((clean_tokens.float(), -clean_tokens.float()), dim=-1)
        reveal = ~valid_tokens
        return state, reveal


def test_renoising_predicts_clean_then_uses_next_forward_noise_level():
    model = RenoisingModel()
    result = run_sampler(
        RenoisingSampler(), model, SamplingRequest(num_samples=1, num_steps=3)
    )

    assert torch.equal(result.samples, torch.full((1, 3), 2))
    assert len(model.forward_noise) == 2
    assert model.forward_noise[0][0] < 0.999
    assert model.forward_noise[1][0] < model.forward_noise[0][0]
    assert result.metadata["nfe"] == 3


def test_renoising_preserves_prompt_tokens():
    model = RenoisingModel()
    request = SamplingRequest(
        num_samples=1,
        num_steps=2,
        prompt_tokens=torch.tensor([[4, 0, 0]]),
        prompt_mask=torch.tensor([[True, False, False]]),
    )

    result = run_sampler(RenoisingSampler(), model, request)

    assert torch.equal(result.samples, torch.tensor([[4, 2, 2]]))
