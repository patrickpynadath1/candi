"""Every algorithm reduces the same way and feeds the sampler that drew its times."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
import torch
from torch import nn

from core.algorithms.training_loss import finish_training_loss
from core.modifiers.token_feedback import TokenFeedback


class RecordingSampler:
    def __init__(self):
        self.observed = []

    def sample(self, count, device, **context):
        return torch.full((count,), 0.5, device=device)

    def observe(self, times, token_loss, mask, step):
        self.observed.append(times.clone())


def mdlm():
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config", overrides=["+experiment=smoke", "algorithm.length=7"]
        )
    return instantiate(cfg.algorithm)


def test_finish_reduces_and_observes_only_in_training():
    per_token = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    mask = torch.tensor([[True, False], [True, True]])
    sampler = RecordingSampler()
    common = dict(tokens=None, prediction=None, step=0, time_sampler=sampler)
    result = finish_training_loss(
        per_token, mask, training=False, sampler_time=torch.ones(2), **common
    )
    assert result["loss"] == 13 / 3 and result["num_tokens"] == 3
    assert not sampler.observed
    finish_training_loss(per_token, mask, training=True, sampler_time=torch.ones(2), **common)
    assert len(sampler.observed) == 1
    with pytest.raises(ValueError, match="at least one target"):
        finish_training_loss(
            per_token, torch.zeros_like(mask), training=True, sampler_time=None, **common
        )


def test_mdlm_feeds_the_per_call_sampler_not_the_default():
    algorithm = mdlm()
    default, override = RecordingSampler(), RecordingSampler()
    algorithm.time_sampler = default
    tokens = torch.randint(0, 10, (2, 7))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    algorithm.loss(tokens, mask, training=True, time_sampler=override)
    assert len(override.observed) == 1 and not default.observed


def test_token_feedback_refuses_to_replace_another_block_policy():
    algorithm = mdlm()

    class OtherPolicy(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("unused")

    algorithm.model.block_execution = OtherPolicy()
    with pytest.raises(ValueError, match="standard block policy"):
        TokenFeedback(algorithm, depth=0)
