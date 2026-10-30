"""Finite birth integration and shared token/position policies on edited layouts."""
import pytest
import torch

from core.noise_schedules.insertion import PowerInsertionSchedule
from core.samplers.insertion import InsertionSampler
from core.samplers.policies import EntropyPositions, RandomPositions, TokenSelection
from core.tests.test_variable_length import assembly


@pytest.mark.parametrize("power", [1., 2., 4.])
@pytest.mark.parametrize("integration", ["finite", "integrated_hazard"])
def test_birth_mean_matches_conditional_survival(power, integration, monkeypatch):
    algo = assembly()
    algo.schedule = PowerInsertionSchedule(power)
    rates, counts = [], []
    predict = algo.predict

    def record_prediction(*args, **kwargs):
        prediction = predict(*args, **kwargs)
        counts.append(prediction.expected_counts.clone())
        return prediction

    def record_rate(rate):
        rates.append(rate.clone())
        return torch.zeros_like(rate)

    monkeypatch.setattr(algo, "predict", record_prediction)
    monkeypatch.setattr(torch, "poisson", record_rate)
    sampler = InsertionSampler(steps=3, epsilon=.01, count_integration=integration)
    algo.generate(sampler, torch.tensor([[1]]), torch.tensor([[True]]))
    times = torch.linspace(1, .01, 4)
    for index, (t, s) in enumerate(zip(times[:-1], times[1:])):
        # Conditional probability that an as-yet absent token is born by s.
        ratio = (1-algo.schedule.survival(s))/(1-algo.schedule.survival(t))
        fraction = 1-ratio if integration == "finite" else -ratio.log()
        torch.testing.assert_close(rates[index][:, -1], counts[index][:, -1]*fraction)


@pytest.mark.parametrize("power", [1., 2., 4.])
def test_conditional_birth_fraction_agrees_with_oracle_simulation(power):
    torch.manual_seed(413)
    schedule = PowerInsertionSchedule(power)
    birth = schedule.birth_times((150000,), "cpu")
    t, s = torch.tensor(.6), torch.tensor(.2)
    absent = birth < t
    observed = ((birth >= s) & absent).sum() / absent.sum()
    expected = -torch.expm1(-schedule.integrated_rate(t, s))
    assert observed.item() == pytest.approx(expected.item(), abs=.006)
    assert schedule.integrated_rate(t, t).item() == 0
    assert -torch.expm1(-schedule.integrated_rate(t, torch.tensor(1e-6))) > .9999


@pytest.mark.parametrize("positions", [EntropyPositions(.1), RandomPositions(1)])
@pytest.mark.parametrize("continuous", [False, True])
def test_shared_policies_preserve_prompts_feedback_and_frozen_tokens(positions, continuous):
    torch.manual_seed(76)
    algo = assembly(feedback=True, continuous=continuous)
    prompt = torch.tensor([[1, 0], [2, 0]])
    fixed = torch.tensor([[True, False], [True, False]])
    sampler = InsertionSampler(steps=4, max_length=20, overflow="truncate",
        tokens=TokenSelection("greedy"), positions=positions, commitment="freeze")
    tokens, info = algo.generate(sampler, prompt, fixed)
    assert torch.equal(tokens[:, 0], prompt[:, 0])
    assert tokens.shape[1] == max(info["lengths"])
    assert tokens.min() >= 0 and tokens.max() < 5
    assert info["nfe"] == 5


def test_commitments_require_clean_token_policy():
    with pytest.raises(ValueError, match="commitments require"):
        InsertionSampler(commitment="freeze")


def test_frozen_token_survives_later_insertions_and_terminal_decode(monkeypatch):
    from core.modifiers.variable_length.model import SequencePrediction
    algo = assembly(feedback=True)
    call = 0

    def prediction(states, points, time, valid, **kwargs):
        nonlocal call
        # Every prediction prefers a new token; the terminal pass must not
        # overwrite the token committed before the second insertion.
        call += 1
        probs = torch.full((*valid.shape, 5), .025)
        probs[..., min(call, 4)] = .9
        return SequencePrediction(probs.log(), torch.ones(len(valid), valid.shape[1]+1),
                                  None, torch.zeros(*valid.shape, 16))

    draws = 0

    def births(rate):
        nonlocal draws
        draws += 1
        result = torch.zeros_like(rate)
        if draws <= 2:
            result[:, -1] = 1
        return result

    monkeypatch.setattr(algo, "predict", prediction)
    monkeypatch.setattr(torch, "poisson", births)
    sampler = InsertionSampler(steps=3, tokens=TokenSelection("greedy"),
                               commitment="freeze", final_denoise="greedy")
    output, _ = algo.generate(sampler, torch.tensor([[0]]), torch.tensor([[True]]))
    assert output.tolist() == [[0, 2, 3]]


def test_greedy_token_policy_changes_native_trajectory(monkeypatch):
    # Record the clean estimate consumed by the native Gaussian update.
    algo = assembly(continuous=True)
    seen = []
    update = algo.views[0].update
    def record(process, representation, state, posterior, source, destination):
        seen.append(posterior.clone())
        return update(process, representation, state, posterior, source, destination)
    monkeypatch.setattr(algo.views[0], "update", record)
    # Append a token on the first interval, so the second interval has an
    # editable token whose native update receives the selected clean estimate.
    def births(rate):
        result = torch.zeros_like(rate)
        result[:, -1] = 1
        return result
    monkeypatch.setattr(torch, "poisson", births)
    sampler = InsertionSampler(steps=2, tokens=TokenSelection("greedy"))
    algo.generate(sampler, torch.tensor([[0]]), torch.tensor([[True]]))
    assert seen[-1].shape[1] == 2
    assert ((seen[-1] == 0) | (seen[-1] == 1)).all()
    torch.testing.assert_close(seen[-1].sum(-1), torch.ones_like(seen[-1][..., 0]))


def test_uniform_confidence_policy_with_noise_births_and_feedback():
    from core.tests.test_variable_length import masked_assembly
    from core.modifiers.variable_length.training import VariableLengthDiffusion
    from core.modifiers.hybrid.views import DiffusionView
    from core.processes.uniform_posterior import PosteriorUniformProcess
    from core.noise_schedules.log_linear import LogLinear
    from core.representations.noisy_inputs import TokenRepresentation
    from core.samplers.hybrid_updates import DiscretePosteriorUpdate

    masked = masked_assembly(head=False, feedback=True)
    masked.algorithm.model.views = torch.nn.ModuleList([DiffusionView(
        PosteriorUniformProcess(5), LogLinear(0), TokenRepresentation(5, 16),
        DiscretePosteriorUpdate())])
    algo = VariableLengthDiffusion(masked.algorithm, 5, predict_gap_tokens=False)
    prompt = torch.tensor([[1, 0], [2, 0]])
    fixed = torch.tensor([[True, False], [True, False]])
    sampler = InsertionSampler(steps=4, max_length=20, overflow="truncate",
        tokens=TokenSelection("greedy"), positions=EntropyPositions(.1), commitment="freeze")
    output, info = algo.generate(sampler, prompt, fixed)
    assert torch.equal(output[:, 0], prompt[:, 0])
    assert info["nfe"] == 5
    assert output.min() >= 0 and output.max() < 5


def test_keep_rate_hazard_matches_birth_conditional_mask_probability():
    from core.objectives.weighting import KeepRateHazardWeighting, ObjectiveContext
    # Nonlinear keep schedule checks the derivative, not merely a hardcoded 1/t.
    def schedule(t):
        return (1-t).square(), -2*(1-t)
    t = torch.tensor([.2, .6], dtype=torch.float64, requires_grad=True)
    birth = torch.tensor([.8, .9], dtype=torch.float64)
    keep, _ = schedule(t)
    born_keep, _ = schedule(birth)
    mask_probability = (1-keep)/(1-born_keep)
    oracle = torch.autograd.grad(mask_probability.log().sum(), t)[0]
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    ce = -logits.log_softmax(-1)[..., 0]
    actual = KeepRateHazardWeighting(schedule)(ce, ObjectiveContext(time=t))
    torch.testing.assert_close(actual, ce * oracle[:, None])
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_flexmdm_style_composes_hazard_and_count_without_gap_head():
    from hydra.utils import instantiate
    from core.sweeps.configuration import compose
    cfg = compose("flexmdm_style")
    assert not cfg.variable_length.predict_gap_tokens
    assert cfg.variable_length.objective.content_weight == 0
    # Match the recipe's standalone weighting against the reference linear 1/t.
    weighting = instantiate(cfg.objective.weighting)
    from core.objectives.weighting import ObjectiveContext
    t = torch.tensor([.25, .75])
    torch.testing.assert_close(weighting(torch.ones(2, 3), ObjectiveContext(time=t)),
                               (1/t)[:, None].expand(2, 3))
