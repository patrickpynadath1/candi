import torch

from samplers.base import SamplingRequest
from samplers.cadd import CADDSampler
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


class CADDModel(ModelBase):
    sampling_family = "cadd"
    sampling_capabilities = frozenset({
        "continuous_prior", "cadd_continuous_transition",
        "cadd_discrete_transition",
    })
    mask_index = 3

    def __init__(self):
        self.final_flags = []

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sample_cadd_continuous_transition(self, *, xt_embeds, xt_tokens, **_kwargs):
        return xt_embeds, torch.full_like(xt_tokens, 2)

    def sample_cadd_discrete_transition(
        self, *, xt_embeds, xt_tokens, x0_hat_tokens, prev_clean_mask,
        noise_removal_step, **_kwargs
    ):
        self.final_flags.append(noise_removal_step)
        tokens = torch.where(prev_clean_mask, xt_tokens, x0_hat_tokens)
        return xt_embeds, tokens, torch.ones_like(prev_clean_mask)


def test_final_step_removes_noise_without_overwriting_prompt():
    model = CADDModel()
    request = SamplingRequest(
        num_samples=1, num_steps=2,
        prompt_tokens=torch.tensor([[1, 0, 0]]),
        prompt_mask=torch.tensor([[1, 0, 0]]),
    )
    result = run_sampler(CADDSampler(), model, request)

    assert result.samples.tolist() == [[1, 2, 2]]
    assert model.final_flags == [False, True]
    assert result.metadata["nfe"] == 2
