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
    model = _bare(CANDI, temp=1.0, neg_infinity=-1e6)
    tokens = torch.tensor([[1, 2]])
    reveal = torch.tensor([[True, False]])
    logits = model._process_model_output(tokens, torch.zeros(1, 2, 4), reveal)
    assert torch.isfinite(logits).all()
    assert logits[0, 0, 1] == 0
    assert logits[0, 0].argmax() == 1
    assert torch.allclose(logits[0, 1].exp().sum(), torch.tensor(1.0))



@pytest.mark.parametrize("schedule", ["linear", "cosine", "geometric"])
def test_candi_sigma_schedules_are_bounded(schedule):
    model = _bare(
        CANDI,
        sigma_min=0.1,
        sigma_max=2.0,
        cont_schedule_target="sigma",
        cont_schedule_type=schedule,
    )
    values = model.get_continuous_from_discrete_noise(torch.tensor([0.0, 0.5, 1.0]))
    assert values.shape == (3,)
    assert torch.isfinite(values).all()
    assert values[0] == pytest.approx(0.1)
    assert values[-1] == pytest.approx(2.0)


def test_candi_discrete_noising_endpoints_and_bos():
    model = _bare(CANDI, vocab_size=5, ignore_bos=True)
    tokens = torch.tensor([[1, 2, 3]])
    assert torch.equal(model.discrete_noising(tokens, torch.ones(1, 1)), tokens)
    noisy = model.discrete_noising(tokens, torch.zeros(1, 1))
    assert noisy[0, 0] == tokens[0, 0]
    assert noisy.min() >= 0 and noisy.max() < 5


def test_candi_q_xt_shapes_and_finite_values():
    model = _bare(
        CANDI,
        vocab_size=5,
        ignore_bos=False,
        sigma_min=0.1,
        sigma_max=1.0,
        cont_schedule_target="sigma",
        cont_schedule_type="linear",
    )
    tokens = torch.tensor([[0, 1, 2], [2, 3, 1]])
    result = model.q_xt(tokens, torch.tensor([[0.5], [0.25]]))
    assert result["xt"].shape == (2, 3, 4)
    assert result["reveal_mask"].shape == tokens.shape
    assert result["discrete_noise"].shape == (2,)
    assert result["continuous_noise"].shape == (2,)
    assert torch.isfinite(result["xt"]).all()


def test_candi_q_xt_accepts_independent_rank_degradation():
    model = _bare(
        CANDI,
        vocab_size=5,
        ignore_bos=False,
        sigma_min=0.1,
        sigma_max=1.0,
        cont_schedule_target="sigma",
        cont_schedule_type="linear",
    )
    tokens = torch.tensor([[0, 1], [2, 3]])
    rank_noise = torch.tensor([0.2, 0.8])
    result = model.q_xt(
        tokens,
        torch.tensor([[0.9], [0.6]]),
        continuous_discrete_noise=rank_noise,
    )
    assert torch.allclose(result["discrete_noise"], torch.tensor([0.1, 0.4]))
    assert torch.equal(result["rank_noise"], rank_noise)
    assert torch.allclose(
        result["continuous_noise"], torch.tensor([0.28, 0.82])
    )


def test_candi_q_xt_keeps_prompt_as_clean_onehot():
    model = _bare(
        CANDI, vocab_size=5, ignore_bos=False, sigma_min=0.1, sigma_max=1.0,
        cont_schedule_target="sigma", cont_schedule_type="linear",
    )
    tokens = torch.tensor([[0, 1, 2, 3]])
    valid = torch.tensor([[0, 0, 1, 1]])
    result = model.q_xt(tokens, torch.zeros(1, 1), valid_tokens=valid)
    expected = torch.nn.functional.one_hot(tokens[:, :2], 4).float()
    assert torch.equal(result["xt"][:, :2], expected)
    assert torch.equal(result["reveal_mask"][:, :2], torch.ones(1, 2))


def test_candi_rejects_unknown_schedule():
    model = _bare(
        CANDI, cont_schedule_target="sigma", cont_schedule_type="unknown"
    )
    with pytest.raises(ValueError, match="Unknown cont_schedule_type"):
        model.get_continuous_from_discrete_noise(torch.tensor([0.5]))


def test_candi_optimized_discrete_step_preserves_clean_prompt():
    model = _bare(CANDI)
    tokens = torch.tensor([[1, 0, 3]])
    predictions = torch.tensor([[2, 2, 2]])
    clean = torch.tensor([[True, False, True]])
    updated, new_clean = model._discrete_step_optimized(
        predictions, tokens.clone(), torch.tensor(0.8), 0.2, clean
    )
    assert torch.equal(updated[clean], tokens[clean])
    assert updated[0, 1] == 2
    assert torch.all(new_clean >= clean)
