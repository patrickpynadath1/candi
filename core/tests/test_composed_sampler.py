"""Choices can be combined without changing the sampler loop."""

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from core.samplers.composed import ComposedSampler
from core.samplers.policies import (
    TokenSelection,
    RandomPositions,
    ForwardCorruption,
    ReverseTransition,
    EntropyPositions,
    ScheduledRandomPositions,
)
from core.tests.test_sampler_equivalence import build_algorithm


@pytest.mark.parametrize("transition", [ForwardCorruption, ReverseTransition])
@pytest.mark.parametrize("commitment", ["none", "freeze", "revise"])
@pytest.mark.parametrize("mode", ["sample", "greedy"])
def test_random_selection_composes_with_updates_and_commitments(
    transition, commitment, mode
):
    algo = build_algorithm("gaussian", "uniform", "posterior_mean")
    sampler = ComposedSampler(
        steps=4,
        tokens=TokenSelection(mode),
        positions=RandomPositions(count=1),
        transition=transition(),
        commitment=commitment,
    )
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    prompt = torch.tensor([[True] * 4, [True, False, False, False]])
    samples, info = algo.generate(sampler, tokens, prompt)
    assert torch.equal(samples[prompt], tokens[prompt])
    assert ((samples >= 0) & (samples < 12)).all()
    if commitment == "none":
        assert info["nfe"] == 5
    else:
        assert info["nfe_per_sample"][0] == 0
        assert all(row == [0, 1] for row in info["accepted_per_step"])
        if commitment == "freeze":
            assert info["nfe"] == 3
            assert info["completed_per_sample"] == [True, True]
        else:
            assert info["nfe"] == 5
            assert info["completed_per_sample"] == [True, False]


def test_random_positions_are_reproducible_and_only_choose_active_positions():
    selection = RandomPositions(count=2)
    active = torch.tensor([[False] * 5, [True, False, True, True, True]])
    p = torch.ones(2, 5, 12) / 12
    torch.manual_seed(17)
    a = selection(p, active)
    torch.manual_seed(17)
    b = selection(p, active)
    assert torch.equal(a, b)
    assert not (a & ~active).any()
    assert a.sum(-1).tolist() == [0, 2]


def test_composed_config_instantiates_and_runs():
    config = OmegaConf.load("configs/inference/sampler/composed.yaml")
    config.steps = 2
    sampler = instantiate(config)
    algo = build_algorithm("masked", None, "none")
    samples, info = algo.generate(
        sampler, torch.ones(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.bool)
    )
    assert samples.shape == (2, 4) and info["nfe"] == 3


def test_invalid_combinations_fail_early():
    with pytest.raises(ValueError, match="require sample or greedy"):
        ComposedSampler(transition=ForwardCorruption())
    with pytest.raises(ValueError, match="require sample or greedy"):
        ComposedSampler(commitment="freeze")
    with pytest.raises(ValueError):
        RandomPositions(0)
    with pytest.raises(ValueError):
        EntropyPositions(float("nan"))


def test_scheduled_random_positions_follow_the_anchoring_schedule():
    # Linear power: after step k of S, round(L * k / S) generated positions are anchored;
    # the last step anchors the rest. Prompts and inactive positions are never chosen.
    policy = ScheduledRandomPositions(power=1.0)
    generated = torch.tensor([[False] * 2 + [True] * 10, [True] * 12])
    committed = ~generated
    probabilities = torch.ones(2, 12, 5) / 5
    torch.manual_seed(3)
    counts = []
    for step in range(4):
        active = generated & ~committed
        selected = policy(probabilities, active, step=step, steps=4, generated=generated)
        assert not (selected & ~active).any()
        committed |= selected
        counts.append((committed & generated).sum(-1).tolist())
    assert counts == [[3, 3], [5, 6], [8, 9], [10, 12]]  # half rounds up
    late = ScheduledRandomPositions(power=3.0)
    first = late(probabilities, generated, step=0, steps=4, generated=generated)
    assert first.sum(-1).tolist() == [0, 0]  # (1/4)^3 of 10 or 12 rounds to zero
    with pytest.raises(ValueError, match="power"):
        ScheduledRandomPositions(0)


@pytest.mark.parametrize("family", ["gaussian", "simplex", "spherical", "vmf"])
@pytest.mark.parametrize("transition", [ReverseTransition, ForwardCorruption])
@pytest.mark.parametrize("mode", ["sample", "greedy"])
def test_random_anchoring_runs_on_pure_continuous_recipes(family, transition, mode):
    algo = build_algorithm(family, None, "none")
    sampler = ComposedSampler(
        steps=5,
        tokens=TokenSelection(mode),
        positions=ScheduledRandomPositions(power=1.5),
        transition=transition(),
        commitment="freeze",
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    prompt = torch.tensor([[True, True] + [False] * 4, [True] + [False] * 5])
    samples, info = algo.generate(sampler, tokens, prompt)
    assert torch.equal(samples[prompt], tokens[prompt])
    # Every position is anchored by the last step, so no terminal call is needed.
    assert info["completed_per_sample"] == [True, True]
    assert info["nfe"] == 5 <= sampler.steps_for_nfe(6) + 1
    assert [sum(row) for row in zip(*info["accepted_per_step"])] == [4, 5]


def test_anchoring_policies_instantiate_from_their_config():
    policies = OmegaConf.load("configs/inference/inference_policies/anchoring.yaml")
    for name, policy in policies.preparation.extra_inference.items():
        sampler = instantiate(policy.sampler)
        assert isinstance(sampler.positions, ScheduledRandomPositions), name
        assert sampler.commitment == "freeze" and sampler.steps_for_nfe(128) == 127
