from types import SimpleNamespace

import torch

from samplers.autoregressive_cached import CachedAutoregressiveSampler
from samplers.base import SamplingRequest
from samplers.runner import run_sampler

from .helpers import ModelBase


class CachedARModel(ModelBase):
    sampling_family = "autoregressive"
    sampling_capabilities = frozenset({"cached_next_token_logits"})
    num_tokens = 5

    def __init__(self):
        self.widths = []

    def next_token_logits_cached(self, tokens, cache=None):
        self.widths.append(tokens.shape[1])
        length = tokens.shape[1] if cache is None else cache.length + tokens.shape[1]
        cache = SimpleNamespace(length=length)
        logits = torch.zeros(*tokens.shape, self.vocab_size)
        logits[..., 2] = 100
        return logits, cache


def test_cached_ar_prefills_once_then_processes_one_token_at_a_time():
    model = CachedARModel()
    result = run_sampler(
        CachedAutoregressiveSampler(use_float64=False, temperature=1),
        model,
        SamplingRequest(num_samples=2),
    )
    assert model.widths == [1, 1, 1, 1]
    assert result.samples[:, 1:].eq(2).all()
    assert result.metadata == {
        "nfe": 4,
        "tokens_processed": 4,
        "cache_used": True,
    }


def test_cached_ar_prefills_the_entire_prompt_once():
    model = CachedARModel()
    request = SamplingRequest(
        num_samples=1,
        prompt_tokens=torch.tensor([[1, 3, 3, 0, 0]]),
        prompt_mask=torch.tensor([[1, 1, 1, 0, 0]]),
    )
    result = run_sampler(
        CachedAutoregressiveSampler(use_float64=False, temperature=1), model, request
    )
    assert model.widths == [3, 1]
    assert result.samples[0, :3].tolist() == [1, 3, 3]
    assert result.samples[0, 3:].tolist() == [2, 2]


def test_cached_ar_rejects_ragged_prompts():
    model = CachedARModel()
    request = SamplingRequest(
        num_samples=2,
        prompt_tokens=torch.tensor([[1, 3, 0, 0, 0], [1, 3, 3, 0, 0]]),
        prompt_mask=torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]),
    )
    try:
        run_sampler(
            CachedAutoregressiveSampler(use_float64=False, temperature=1),
            model,
            request,
        )
    except ValueError as error:
        assert "equal prompt lengths" in str(error)
    else:
        raise AssertionError("expected ragged prompts to be rejected")
