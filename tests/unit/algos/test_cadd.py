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


def test_model_output_is_normalized_and_preserves_revealed_tokens():
    model = _bare(CADD, temp=1.0, neg_infinity=-1e6)
    tokens = torch.tensor([[1, 2]])
    reveal = torch.tensor([[True, False]])
    logits = model._process_model_output(tokens, torch.zeros(1, 2, 4), reveal)
    assert torch.isfinite(logits).all()
    assert logits[0, 0, 1] == 0
    assert logits[0, 0].argmax() == 1
    assert torch.allclose(logits[0, 1].exp().sum(), torch.tensor(1.0))



def test_cadd_discrete_noising_and_q_xt_shapes():
    class Backbone(torch.nn.Module):
        def get_embedding(self, tokens):
            return torch.nn.functional.one_hot(tokens, 4).float()

        def get_mask_embed(self):
            return torch.ones(4)

    model = _bare(
        CADD, backbone=Backbone(), mask_index=4, ignore_bos=True
    )
    tokens = torch.tensor([[1, 2, 3]])
    assert torch.equal(model.discrete_noising(tokens, torch.ones(1, 1)), tokens)
    result = model.q_xt(tokens, torch.zeros(1, 1))
    assert result["xt"].shape == (1, 3, 4)
    assert result["xt_tokens"].shape == tokens.shape
    assert result["reveal_mask"].shape == tokens.shape
    assert result["xt_tokens"][0, 0] == 1
    assert torch.isfinite(result["xt"]).all()


def test_cadd_mask_frequency_matches_requested_transition():
    torch.manual_seed(7)
    alpha = 0.7
    model = _bare(CADD, mask_index=4, ignore_bos=False)
    tokens = torch.zeros(40_000, 1, dtype=torch.long)
    noisy = model.discrete_noising(tokens, torch.tensor([[alpha]]))
    assert noisy.eq(model.mask_index).float().mean().item() == pytest.approx(
        1 - alpha, abs=0.01
    )


def test_cadd_nll_matches_independent_formula():
    model = _bare(CADD)
    probabilities = torch.tensor([[[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]])
    targets = torch.tensor([[1, 0]])
    alpha = torch.tensor([[0.6]])
    derivative = torch.tensor([[-0.8]])
    result = model.nll_per_token(
        probabilities.log(), targets, alpha, derivative, reveal_mask=None
    )
    expected = torch.tensor([[0.7, 0.6]]).log() * derivative / (1 - alpha)
    assert torch.allclose(result, expected)


def test_cadd_prompt_injection_preserves_prompt_embeddings():
    model = _bare(CADD)
    x = torch.randn(1, 3, 4)
    prompt = torch.tensor([[2, 0, 1]])
    mask = torch.tensor([[1, 0, 1]])
    result, clean = model.generate_sample_prompt(prompt, x, mask)
    expected = torch.nn.functional.one_hot(prompt, 4).float()
    assert torch.equal(result[mask.bool()], expected[mask.bool()])
    assert torch.equal(result[~mask.bool()], x[~mask.bool()])
    assert torch.equal(clean, mask.bool())


def test_cadd_generation_preserves_prompt_tokens(monkeypatch):
    model = _bare(
        CADD,
        num_tokens=3,
        mask_index=4,
        config=SimpleNamespace(
            model=SimpleNamespace(hidden_size=4),
            sampling=SimpleNamespace(steps=1),
        ),
    )
    model.prior_sample = lambda *shape: torch.zeros(*shape, 4)
    monkeypatch.setattr(CADD, "device", property(lambda _self: torch.device("cpu")), raising=False)
    model._continuous_step = lambda **kwargs: (
        kwargs["xt_embeds"], torch.full(kwargs["xt_tokens"].shape, 2)
    )
    prompt = torch.tensor([[1, 0, 3]])
    prompt_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    result = model.generate_samples(
        1, num_steps=1, prompt_tokens=prompt, prompt_mask=prompt_mask
    )
    assert torch.equal(result[prompt_mask], prompt[prompt_mask])
    assert result[0, 1] == 2
