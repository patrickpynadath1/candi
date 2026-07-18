import torch

from samplers.base import SamplingRequest
from samplers.continuous_embedding import ContinuousEmbeddingSampler
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


class EmbeddingModel(ModelBase):
    sampling_family = "continuous_embedding"
    sampling_capabilities = frozenset({
        "continuous_prior", "continuous_schedule", "continuous_transition",
    })

    def __init__(self):
        self.clean_masks = []

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sampling_continuous_schedule(self, times):
        return times.square()

    def sample_continuous_transition(self, *, x, clean_mask, **_kwargs):
        self.clean_masks.append(clean_mask.clone())
        prediction = torch.zeros_like(x)
        prediction[..., 3] = 1
        prediction[clean_mask] = x[clean_mask]
        return prediction, prediction


def test_passes_prompt_mask_to_embedding_transition():
    model = EmbeddingModel()
    mask = torch.tensor([[0, 1, 0]])
    request = SamplingRequest(
        num_samples=1, num_steps=1,
        prompt_tokens=torch.tensor([[0, 2, 0]]), prompt_mask=mask,
    )
    result = run_sampler(ContinuousEmbeddingSampler(), model, request)

    assert torch.equal(model.clean_masks[0], mask.bool())
    assert result.samples.tolist() == [[3, 2, 3]]
