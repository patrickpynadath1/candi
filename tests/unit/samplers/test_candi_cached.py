import torch

from samplers.base import SamplingRequest
from samplers.candi_cached import CachedCANDISampler
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


class CachedCANDIModel(ModelBase):
    sampling_family = "candi"
    sampling_capabilities = frozenset({
        "continuous_prior", "continuous_schedule", "embedding_projection",
        "candi_cached_continuous_transition",
        "candi_cached_discrete_transition",
    })

    def __init__(self):
        self.cache_ids = []
        self.first_flags = []

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sampling_continuous_schedule(self, times, _steps):
        return times

    def sampling_embeddings(self, probabilities):
        return probabilities.clone()

    def sample_candi_cached_continuous_transition(
        self, *, x, embedding_cache, is_first, **_kwargs
    ):
        self.cache_ids.append(id(embedding_cache))
        self.first_flags.append(is_first)
        return embedding_cache, torch.full_like(x, 2)

    def sample_candi_cached_discrete_transition(
        self, predicted, _tokens, *_args, prev_clean_mask, **_kwargs
    ):
        return predicted, prev_clean_mask


def test_reuses_embedding_cache_and_marks_only_the_first_step():
    model = CachedCANDIModel()
    result = run_sampler(
        CachedCANDISampler(), model, SamplingRequest(num_samples=1, num_steps=2)
    )

    assert result.samples.tolist() == [[2, 2, 2]]
    assert model.first_flags == [True, False]
    assert len(set(model.cache_ids)) == 1
    assert result.metadata["nfe"] == 2
