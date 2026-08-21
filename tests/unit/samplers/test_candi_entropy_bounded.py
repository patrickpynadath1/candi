from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.candi_entropy_bounded import CANDIEntropyBoundedSampler
from samplers.runner import run_sampler

from .helpers import ModelBase, onehot_prior


class CANDIEntropyModel(ModelBase):
    sampling_family = "candi"
    sampling_capabilities = frozenset({
        "continuous_prior", "continuous_schedule", "candi_discrete_schedule",
        "candi_continuous_transition",
    })
    num_tokens = 3

    def __init__(self):
        self.inputs = []
        self.config = SimpleNamespace(sampling=SimpleNamespace(steps=3))

    def prior_sample(self, batch, length):
        return onehot_prior(batch, length)

    def sampling_continuous_schedule(self, times, steps):
        del steps
        return times

    def sampling_discrete_schedule(self, times):
        return times

    def sample_candi_continuous_transition(self, x, *_args, **_kwargs):
        self.inputs.append(x.clone())
        probs = torch.tensor([
            [0.99, 0.005, 0.005, 0.0],
            [0.80, 0.10, 0.10, 0.0],
            [0.55, 0.225, 0.225, 0.0],
        ]).expand(x.shape[0], -1, -1)
        return x + 0.1, probs


def test_rejected_candi_sites_continue_continuous_trajectory():
    model = CANDIEntropyModel()
    result = run_sampler(
        CANDIEntropyBoundedSampler(0.0), model,
        SamplingRequest(num_samples=1, num_steps=3),
    )

    assert result.metadata["accepted_per_step"] == [[1], [1], [1]]
    # The first coordinate was replaced by a clean one-hot. Rejected
    # coordinates carry the +0.1 continuous solver update into pass two.
    assert torch.isclose(model.inputs[1][0, 0].sum(), torch.tensor(1.0))
    assert torch.allclose(
        model.inputs[1][0, 1:], model.inputs[0][0, 1:] + 0.1
    )


def test_large_bound_accepts_candi_sites_in_one_pass():
    result = run_sampler(
        CANDIEntropyBoundedSampler(10.0), CANDIEntropyModel(),
        SamplingRequest(num_samples=2, num_steps=3),
    )
    assert result.metadata["nfe"] == 1
    assert result.metadata["nfe_per_sample"] == [1, 1]


def test_greedy_candi_eb_commits_argmax_tokens():
    result = run_sampler(
        CANDIEntropyBoundedSampler(10.0, greedy_clean_estimate=True),
        CANDIEntropyModel(),
        SamplingRequest(num_samples=2, num_steps=3),
    )
    assert torch.equal(result.samples, torch.zeros_like(result.samples))
    assert result.metadata["greedy_clean_estimate"] is True
