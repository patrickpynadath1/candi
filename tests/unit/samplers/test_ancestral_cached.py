import torch

from samplers.ancestral_cached import CachedAncestralSampler
from samplers.base import SamplingRequest
from samplers.runner import run_sampler

from .helpers import ModelBase


class CachedModel(ModelBase):
    sampling_family = "absorbing_diffusion"
    sampling_capabilities = frozenset({
        "discrete_prior", "noise_schedule", "ancestral_transition",
    })
    time_conditioning = False

    def __init__(self):
        self.seen_caches = []
        self.seen_prompt_masks = []

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, dtype=torch.long)

    def noise(self, time):
        return None, time

    def sample_ancestral_transition(self, *, x, p_x0, prompt_mask=None, **_kwargs):
        self.seen_caches.append(p_x0)
        self.seen_prompt_masks.append(prompt_mask)
        return "cached-probs", x


def test_reuses_cache_when_tokens_and_conditioning_are_unchanged():
    model = CachedModel()
    result = run_sampler(
        CachedAncestralSampler("none"),
        model,
        SamplingRequest(num_samples=1, num_steps=2),
    )

    assert model.seen_caches == [None, "cached-probs"]
    assert result.metadata["nfe"] == 2


def test_cached_ancestral_forwards_prompt_mask_on_every_transition():
    model = CachedModel()
    mask = torch.tensor([[1, 0, 1]])
    run_sampler(
        CachedAncestralSampler("none"),
        model,
        SamplingRequest(
            num_samples=1,
            num_steps=2,
            prompt_tokens=torch.tensor([[3, 0, 2]]),
            prompt_mask=mask,
        ),
    )
    assert all(torch.equal(seen, mask) for seen in model.seen_prompt_masks)


def test_invalidates_cache_for_time_conditioned_models():
    model = CachedModel()
    model.time_conditioning = True
    run_sampler(
        CachedAncestralSampler("none"),
        model,
        SamplingRequest(num_samples=1, num_steps=2),
    )

    assert model.seen_caches == [None, None]
