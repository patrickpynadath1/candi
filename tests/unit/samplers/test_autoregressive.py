import torch
import pytest

from samplers.autoregressive import AutoregressiveSampler
from samplers.base import SamplingRequest
from samplers.runner import run_sampler

from .helpers import ModelBase


class ARModel(ModelBase):
    sampling_family = "autoregressive"
    sampling_capabilities = frozenset({"next_token_logits"})
    num_tokens = 4

    def next_token_logits(self, tokens):
        logits = torch.zeros(*tokens.shape, self.vocab_size)
        logits[..., 2] = 100
        return logits


def test_generates_left_to_right_and_reports_model_evaluations():
    result = run_sampler(
        AutoregressiveSampler(use_float64=False, temperature=1),
        ARModel(),
        SamplingRequest(num_samples=2),
    )

    assert torch.equal(result.samples[:, 0], torch.ones(2, dtype=torch.long))
    assert torch.equal(result.samples[:, 1:], torch.full((2, 3), 2))
    assert result.metadata == {"nfe": 3}


def test_preserves_a_fixed_prompt():
    request = SamplingRequest(
        num_samples=1,
        prompt_tokens=torch.tensor([[1, 3, 0, 0]]),
        prompt_mask=torch.tensor([[1, 1, 0, 0]]),
    )
    result = run_sampler(
        AutoregressiveSampler(use_float64=False, temperature=1), ARModel(), request
    )

    assert result.samples[0, :2].tolist() == [1, 3]
    assert result.samples[0, 2:].tolist() == [2, 2]


def test_fully_prompted_sequence_skips_model_evaluation():
    request = SamplingRequest(
        num_samples=1,
        prompt_tokens=torch.tensor([[1, 3, 2, 2]]),
        prompt_mask=torch.ones(1, 4),
    )
    result = run_sampler(
        AutoregressiveSampler(use_float64=False, temperature=1), ARModel(), request
    )
    assert result.samples.tolist() == [[1, 3, 2, 2]]
    assert result.metadata["nfe"] == 0


@pytest.mark.parametrize(
    "mask",
    [torch.tensor([[1, 0, 1, 0]]), torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])],
)
def test_rejects_prompts_that_are_not_a_uniform_prefix(mask):
    tokens = torch.zeros_like(mask)
    with pytest.raises(ValueError, match="contiguous prefixes|equal prompt lengths"):
        run_sampler(
            AutoregressiveSampler(use_float64=False, temperature=1),
            ARModel(),
            SamplingRequest(
                num_samples=mask.shape[0], prompt_tokens=tokens, prompt_mask=mask
            ),
        )
