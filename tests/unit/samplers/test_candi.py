import torch
import pytest

from samplers.base import SamplingRequest
from samplers.candi import CANDISampler
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


@pytest.mark.parametrize("greedy", [False, True])
def test_greedy_changes_only_token_commit_distribution(greedy):
    class RecordingModel(CANDIModel):
        def sample_candi_continuous_transition(self, x, *_args, **_kwargs):
            self.continued = x + 0.25
            self.probs = torch.zeros_like(x)
            self.probs[..., 1] = 0.4
            self.probs[..., 2] = 0.6
            return self.continued, self.probs

        def sample_candi_discrete_transition(
            self, continuous, probs, time, dt, *, prev_clean_mask
        ):
            assert torch.equal(continuous, self.continued)
            expected = torch.zeros_like(probs)
            expected[..., 2] = 1
            assert torch.equal(probs, expected if greedy else self.probs)
            assert time > 0 and dt > 0
            return continuous, prev_clean_mask

    result = run_sampler(
        CANDISampler(greedy_clean_estimate=greedy), RecordingModel(),
        SamplingRequest(num_samples=2, num_steps=2),
    )
    assert result.metadata["greedy_clean_estimate"] is greedy


class CANDIModel(ModelBase):
    sampling_family = "candi"
    sampling_capabilities = frozenset({
        "continuous_prior", "continuous_schedule", "candi_discrete_schedule",
        "candi_continuous_transition", "candi_discrete_transition",
    })

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sampling_continuous_schedule(self, times, steps):
        assert len(times) == steps + 1
        return times

    def sampling_discrete_schedule(self, times):
        return times

    def sample_candi_continuous_transition(self, x, *_args, **_kwargs):
        prediction = torch.zeros_like(x)
        prediction[..., 2] = 1
        return prediction, prediction

    def sample_candi_discrete_transition(
        self, continuous, _clean_probs, _time, _dt, *, prev_clean_mask
    ):
        return continuous, prev_clean_mask


def test_uncached_sampler_alternates_continuous_and_discrete_updates():
    result = run_sampler(
        CANDISampler(), CANDIModel(), SamplingRequest(num_samples=2, num_steps=2)
    )

    assert torch.equal(result.samples, torch.full((2, 3), 2))
    assert result.metadata["nfe"] == 2


class EmbeddedCANDIModel(CANDIModel):
    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, 2)

    def sampling_prompt_state(self, state, prompt_tokens, prompt_mask):
        clean = torch.zeros(state.shape[:2], dtype=torch.bool)
        if prompt_tokens is None or prompt_mask is None:
            return state, clean
        prompt_state = torch.stack(
            (prompt_tokens.float(), -prompt_tokens.float()), dim=-1
        )
        mask = prompt_mask.bool()
        return torch.where(mask.unsqueeze(-1), prompt_state, state), mask

    def sampling_state_tokens(self, state):
        return state[..., 0].long()

    def sample_candi_continuous_transition(self, x, *_args, **_kwargs):
        return x, torch.zeros(*x.shape[:2], 100)


def test_uncached_sampler_uses_embedding_prompt_and_token_hooks():
    request = SamplingRequest(
        num_samples=1,
        num_steps=1,
        prompt_tokens=torch.tensor([[42, 0, 0]]),
        prompt_mask=torch.tensor([[True, False, False]]),
    )

    result = run_sampler(CANDISampler(), EmbeddedCANDIModel(), request)

    assert result.samples[0, 0] == 42


def test_uncached_sampler_traces_argmax_changes_outside_prompt():
    class ChangingModel(CANDIModel):
        def __init__(self):
            self.calls = 0

        def sample_candi_continuous_transition(self, x, *_args, **_kwargs):
            prediction = torch.zeros_like(x)
            prediction[..., 1 + self.calls] = 1
            self.calls += 1
            return prediction, prediction

    request = SamplingRequest(
        num_samples=1,
        num_steps=2,
        prompt_tokens=torch.tensor([[0, 0, 0]]),
        prompt_mask=torch.tensor([[True, False, False]]),
    )
    result = run_sampler(CANDISampler(trace_argmax_changes=True), ChangingModel(), request)

    assert result.metadata["argmax_changes_per_sample"] == [2]
    assert result.metadata["argmax_changed_positions_per_sample"] == [2]
    assert result.metadata["argmax_changes_per_step"] == [[2]]
    assert result.metadata["argmax_comparisons"] == 1
