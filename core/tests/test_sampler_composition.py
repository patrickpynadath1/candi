"""Compare composed sampling against frozen pre-refactor loops and RNG streams."""

from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from core.samplers.discrete import DiscreteSampler
from core.samplers.loop import FixedGridLoop
from core.samplers.continuous import FixedStepSampler
from core.samplers.integration import HeunUpdate
from core.samplers.ancestral import AncestralSampler
from core.samplers.uniform_ancestral import UniformAncestralSampler
from core.samplers.analytic import AnalyticSampler
from core.tests.reference_samplers.ancestral import AncestralSampler as OldAncestral
from core.tests.reference_samplers.uniform_ancestral import (
    UniformAncestralSampler as OldUniform,
)
from core.tests.reference_samplers.analytic import AnalyticSampler as OldAnalytic


@pytest.mark.parametrize(
    "method,current,old",
    [
        ("sudoku_hard", AncestralSampler, OldAncestral),
        ("uniform_diffusion", UniformAncestralSampler, OldUniform),
        ("sedd_absorb", AnalyticSampler, OldAnalytic),
    ],
)
@pytest.mark.parametrize("final", ["none", "ancestral"])
def test_frozen_discrete_loop_equivalence(method, current, old, final):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+experiment={method}",
                "model.hidden_size=32",
                "model.n_heads=4",
                "model.n_blocks=1",
                "model.cond_dim=16",
                "model.dropout=0",
                "algorithm.length=8",
            ],
        )
    torch.manual_seed(31)
    algo = instantiate(cfg.algorithm).eval()
    # Nontrivial fixed tokens and an all-fixed sequence exercise prompt handling.
    prompt = torch.arange(16).reshape(2, 8) % 9
    mask = torch.tensor(
        [[True, False, False, True, False, False, False, False], [True] * 8]
    )
    trace = []
    prediction = algo.sampling_prediction

    def record(state, time):
        trace.append((state.clone(), time.clone()))
        return prediction(state, time)

    algo.sampling_prediction = record
    torch.manual_seed(71)
    expected, expected_stats = old(steps=5, final_denoise=final).sample(
        algo, prompt, mask
    )
    expected_rng, expected_trace = torch.get_rng_state(), trace[:]
    trace.clear()
    torch.manual_seed(71)
    actual, actual_stats = current(steps=5, final_denoise=final).sample(
        algo, prompt, mask
    )
    assert torch.equal(actual, expected)
    assert actual_stats == expected_stats
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert len(trace) == len(expected_trace)
    for actual_args, expected_args in zip(trace, expected_trace):
        for actual_arg, expected_arg in zip(actual_args, expected_args):
            torch.testing.assert_close(actual_arg, expected_arg, rtol=0, atol=0)


def test_composed_discrete_rule_reports_actual_evaluations_and_custom_grid():
    class TwoEvaluationRule:
        prediction_feedback = False

        def initialize(self, algo, prompt, mask):
            return prompt.clone()

        def update(self, algo, state, time, next_time, prompt, mask):
            assert not torch.is_grad_enabled()
            return state + 1, 2

        def finalize(self, algo, state, time, prompt, mask):
            return state, 0

    def grid(device):
        return torch.tensor([1.0, 0.8, 0.2], device=device)

    tokens, stats = DiscreteSampler(TwoEvaluationRule(), time_grid=grid).sample(
        SimpleNamespace(), torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool)
    )
    assert (tokens == 2).all()
    assert stats == {"nfe": 4, "nfe_per_sample": [4, 4]}


def test_multistage_rule_uses_one_feedback_context_per_accepted_step():
    from contextlib import contextmanager

    events = []

    @contextmanager
    def context():
        events.append("enter")
        yield
        events.append("exit")

    def field(time, state):
        events.append("prediction")
        return state

    field.step_context = context
    sampler = FixedStepSampler(steps=2, update_rule=HeunUpdate())
    sampler.integrate(field, torch.ones(1))
    assert events == ["enter", "prediction", "prediction", "exit"] * 2


def test_loop_accepts_nested_hybrid_states_without_interpreting_them():
    initial = (torch.ones(2), torch.zeros(2, dtype=torch.long))
    result = FixedGridLoop()(
        initial,
        torch.tensor([1.0, 0.5, 0.0]),
        lambda state, t, s, index: (state[0] * 2, state[1] + 1),
    )
    assert torch.equal(result[0], torch.full((2,), 4.0))
    assert torch.equal(result[1], torch.full((2,), 2))
