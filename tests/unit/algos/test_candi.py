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


def test_candi_discrete_noising_matches_uniform_transition_probability():
    torch.manual_seed(5)
    vocab_size = 5
    alpha = 0.65
    model = _bare(CANDI, vocab_size=vocab_size, ignore_bos=False)
    tokens = torch.zeros(50_000, 1, dtype=torch.long)

    noisy = model.discrete_noising(tokens, torch.tensor([[alpha]]))

    # A corrupted token is drawn uniformly and can coincidentally equal x0.
    expected_unchanged = alpha + (1 - alpha) / vocab_size
    observed_unchanged = noisy.eq(tokens).float().mean().item()
    assert observed_unchanged == pytest.approx(expected_unchanged, abs=0.01)
    frequencies = torch.bincount(noisy.flatten(), minlength=vocab_size).float()
    frequencies /= noisy.numel()
    assert torch.allclose(
        frequencies[1:],
        torch.full((vocab_size - 1,), (1 - alpha) / vocab_size),
        atol=0.01,
    )


@pytest.mark.parametrize(
    "pure_continuous,weighting",
    [(True, "none"), (False, "none"), (False, "likelihood")],
)
def test_candi_nll_matches_independent_formula(pure_continuous, weighting):
    model = _bare(
        CANDI,
        pure_continuous=pure_continuous,
        hybrid_loss_weighting=weighting,
    )
    probabilities = torch.tensor([[[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]])
    targets = torch.tensor([[1, 0]])
    alpha = torch.tensor([[0.6]])
    derivative = torch.tensor([[-0.8]])

    result = model.nll_per_token(
        probabilities.log(), targets, alpha, derivative, reveal_mask=None
    )
    selected_log_probability = torch.tensor([[0.7, 0.6]]).log()
    expected = -selected_log_probability
    if not pure_continuous and weighting == "likelihood":
        expected = selected_log_probability * derivative / (1 - alpha)
    assert torch.allclose(result, expected)


def test_candi_raw_ce_excludes_likelihood_weighting():
    probabilities = torch.tensor([[[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]])
    targets = torch.tensor([[1, 0]])
    result = CANDI.raw_ce_per_token(probabilities.log(), targets)
    assert torch.allclose(result, -torch.tensor([[0.7, 0.6]]).log())


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


def test_factorized_time_sampling_covers_full_stratified_marginal():
    torch.manual_seed(7)
    values = CANDI._sample_factorized_time(32, torch.device("cpu"), eps=0.01)
    bins = ((values - 0.01) / 0.99 * 32).floor().long()
    assert torch.equal(bins.sort().values, torch.arange(32))
    assert values.min() >= 0.01
    assert values.max() <= 1


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
