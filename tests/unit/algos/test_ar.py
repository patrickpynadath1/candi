# ruff: noqa: F401
from types import SimpleNamespace

import pytest
import torch

from algos.ar import AR
from algos.cadd import CADD
from algos.candi import CANDI
from algos.duo import DUO, DUO_BASE, Integral
from algos.mdlm import MDLM


def _bare(cls, **attrs):
    instance = object.__new__(cls)
    torch.nn.Module.__init__(instance)
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance



def test_ar_processes_shifted_inputs_and_masks():
    ar = _bare(AR)
    tokens = torch.tensor([[1, 2, 3]])
    valid = torch.tensor([[1, 1, 0]])
    inputs, outputs, output_valid = ar._process_model_input(tokens, valid)
    assert torch.equal(inputs, torch.tensor([[1, 2]]))
    assert torch.equal(outputs, torch.tensor([[2, 3]]))
    assert torch.equal(output_valid, torch.tensor([[1, 0]]))


def test_ar_nll_matches_cross_entropy_and_excludes_mask_token():
    class Backbone(torch.nn.Module):
        def forward(self, x, _sigma):
            logits = torch.tensor([1.0, 2.0, 3.0, 50.0])
            return logits.expand(x.shape[0], x.shape[1], -1).clone()

    ar = _bare(AR, backbone=Backbone(), mask_index=3, neg_infinity=-1e6)
    inputs = torch.tensor([[0, 1]])
    targets = torch.tensor([[2, 1]])
    nll = ar.nll(inputs, targets, None)
    expected = -torch.log_softmax(torch.tensor([1.0, 2.0, 3.0, -1e6]), 0)[targets]
    assert torch.allclose(nll, expected)
    assert torch.isfinite(nll).all()


def test_ar_generation_preserves_prompt_and_vocabulary_bounds():
    class Backbone(torch.nn.Module):
        def forward(self, x, _sigma):
            logits = torch.zeros(x.shape[0], x.shape[1], 4)
            logits[..., 2] = 10
            return logits

    ar = _bare(
        AR,
        backbone=Backbone(),
        num_tokens=4,
        vocab_size=4,
        mask_index=3,
        neg_infinity=-1e6,
        temp=1.0,
        config=SimpleNamespace(sampling=SimpleNamespace(use_float64=False)),
        tokenizer=SimpleNamespace(bos_token_id=1),
    )
    ar._device = torch.device("cpu")
    prompt = torch.tensor([[1, 0, 0, 0]])
    mask = torch.tensor([[1, 0, 0, 0]])
    samples = ar.generate_samples(1, prompt_tokens=prompt, prompt_mask=mask)
    assert samples.shape == (1, 4)
    assert samples[0, 0] == 1
    assert samples.min() >= 0 and samples.max() < 4

