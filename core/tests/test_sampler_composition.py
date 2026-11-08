"""Compare composed sampling against frozen pre-refactor loops and RNG streams."""

from types import SimpleNamespace
import torch
from core.samplers.discrete import DiscreteSampler
from core.samplers.loop import FixedGridLoop
from core.samplers.continuous import FixedStepSampler
from core.samplers.integration import HeunUpdate



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
