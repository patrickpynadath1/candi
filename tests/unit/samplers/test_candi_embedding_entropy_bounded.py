from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.candi_embedding_entropy_bounded import (
    CANDIEmbeddingEntropyBoundedSampler,
)
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


class EmbeddingEntropyModel(ModelBase):
    sampling_family = "candi_embedding"
    sampling_capabilities = frozenset({
        "continuous_prior", "continuous_schedule", "continuous_transition",
    })
    num_tokens = 3

    def __init__(self):
        self.inputs = []
        self.noise = SimpleNamespace(alpha_t=lambda times: 1 - times)
        self.config = SimpleNamespace(sampling=SimpleNamespace(steps=3))

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sampling_continuous_schedule(self, times):
        return times

    def sampling_prompt_state(self, state, tokens, mask):
        embeddings = torch.nn.functional.one_hot(tokens, 4).to(state)
        return torch.where(mask.unsqueeze(-1), embeddings, state)

    def sample_continuous_transition(self, state, *_args, **_kwargs):
        self.inputs.append(state.clone())
        probs = torch.tensor([
            [0.99, 0.005, 0.005, 0.0],
            [0.80, 0.10, 0.10, 0.0],
            [0.55, 0.225, 0.225, 0.0],
        ]).expand(state.shape[0], -1, -1)
        return state + 0.1, probs


def test_rejected_embedding_sites_continue_trajectory():
    model = EmbeddingEntropyModel()
    result = run_sampler(
        CANDIEmbeddingEntropyBoundedSampler(0.0, greedy_clean_estimate=True),
        model,
        SamplingRequest(num_samples=1, num_steps=3),
    )

    assert result.metadata["accepted_per_step"] == [[1], [1], [1]]
    assert torch.isclose(model.inputs[1][0, 0].sum(), torch.tensor(1.0))
    assert torch.allclose(
        model.inputs[1][0, 1:], model.inputs[0][0, 1:] + 0.1
    )
    assert result.samples.tolist() == [[0, 0, 0]]


def test_large_bound_accepts_embedding_sites_in_one_pass():
    result = run_sampler(
        CANDIEmbeddingEntropyBoundedSampler(10.0),
        EmbeddingEntropyModel(),
        SamplingRequest(num_samples=2, num_steps=3),
    )
    assert result.metadata["nfe"] == 1
    assert result.metadata["nfe_per_sample"] == [1, 1]
