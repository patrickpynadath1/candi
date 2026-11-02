"""Sequence budgets must include every denoised block and terminal evaluation."""

import pytest
from omegaconf import OmegaConf
from core.samplers.ancestral import AncestralSampler
from core.sweeps.budget import sampler_steps


@pytest.mark.parametrize("length, expected", [(256, 15), (1024, 3), (257, 13)])
def test_block_sequence_budget(length, expected):
    config = OmegaConf.create(
        {
            "task": {"length": length},
            "block_diffusion": {"enabled": True, "block_size": 32},
        }
    )
    assert sampler_steps(config, AncestralSampler(), 128) == expected


def test_unblocked_budget_unchanged():
    assert sampler_steps(OmegaConf.create({}), AncestralSampler(), 128) == 127


def test_insufficient_block_budget_rejected():
    config = OmegaConf.create(
        {
            "task": {"length": 1024},
            "block_diffusion": {"enabled": True, "block_size": 32},
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        sampler_steps(config, AncestralSampler(), 32)
