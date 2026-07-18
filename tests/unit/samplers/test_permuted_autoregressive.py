from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.permuted_autoregressive import PermutedAutoregressiveSampler
from samplers.permuted_autoregressive_uncached import (
    UncachedPermutedAutoregressiveSampler,
)
from samplers.runner import run_sampler

from .helpers import ModelBase


class PermutedModel(ModelBase):
    sampling_family = "permuted_autoregressive"
    sampling_capabilities = frozenset(
        {
            "permuted_logits",
            "cached_permuted_logits",
        }
    )
    num_tokens = 5
    mask_id = 3
    mask_index = 3

    def __init__(self):
        self.cached_widths = []

    @staticmethod
    def _logits(tokens, query_positions):
        logits = torch.zeros(*tokens.shape, 4)
        preferred = (query_positions % 2) + 1
        return logits.scatter(-1, preferred[..., None], 100)

    def permuted_logits(self, tokens, key_position_ids, query_position_ids):
        return self._logits(tokens, query_position_ids)

    def permuted_logits_cached(
        self, tokens, key_position_ids, query_position_ids, cache=None
    ):
        self.cached_widths.append(tokens.shape[1])
        length = tokens.shape[1] if cache is None else cache.length + tokens.shape[1]
        return self._logits(tokens, query_position_ids), SimpleNamespace(length=length)


def test_cached_and_uncached_any_order_sampling_agree():
    order = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    request = SamplingRequest(num_samples=2, options={"order": order})
    model = PermutedModel()
    cached = run_sampler(
        PermutedAutoregressiveSampler(use_float64=False, temperature=1),
        model,
        request,
    )
    uncached = run_sampler(
        UncachedPermutedAutoregressiveSampler(use_float64=False, temperature=1),
        model,
        request,
    )
    assert torch.equal(cached.samples, uncached.samples)
    assert model.cached_widths == [1, 1, 1, 1]
    assert cached.metadata["tokens_processed"] == 4
    assert uncached.metadata["tokens_processed"] == 10
    expected = (torch.arange(1, 5) % 2) + 1
    assert torch.equal(cached.samples[0, 1:], expected)


def test_any_order_rejects_non_permutations():
    request = SamplingRequest(
        num_samples=1, options={"order": torch.tensor([[0, 0, 2, 3]])}
    )
    try:
        run_sampler(
            PermutedAutoregressiveSampler(use_float64=False, temperature=1),
            PermutedModel(),
            request,
        )
    except ValueError as error:
        assert "must be a permutation" in str(error)
    else:
        raise AssertionError("expected invalid order to be rejected")
