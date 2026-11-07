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


def test_latent_budget_is_split_across_latent_blocks():
    from core.samplers.latent_autoencoder import LatentAutoencoderSampler

    config = OmegaConf.create(
        {
            "algorithm": {
                "_target_": "core.algorithms.latent_autoencoder.LatentAutoencoderDiffusion"
            },
            "task": {"length": 256},
            "latent": {"block_size": 32},
        }
    )
    # 8 blocks x (steps + terminal denoise + decode) must fit in 128 calls.
    steps = sampler_steps(config, LatentAutoencoderSampler(), 128)
    assert 8 * (steps + 2) <= 128 and steps == 14


@pytest.mark.parametrize(
    "factory, used",
    [
        ("heun", lambda steps: 2 * steps + 1),
        ("simplex", lambda steps: steps),
        ("simplex_guided", lambda steps: 2 * steps),
        ("spherical", lambda steps: steps),
        ("blur", lambda steps: steps + 1),
        ("blur_bootstrap", lambda steps: 2 * (steps + 1)),
    ],
)
def test_budgeted_samplers_use_the_whole_budget_without_exceeding_it(factory, used):
    from core.samplers.blur_ancestral import BlurAncestralSampler
    from core.samplers.heun import HeunSampler
    from core.samplers.simplex import SimplexBridgeSampler
    from core.samplers.spherical import SphericalFlowSampler

    sampler = {
        "heun": lambda: HeunSampler(),
        "simplex": lambda: SimplexBridgeSampler(),
        "simplex_guided": lambda: SimplexBridgeSampler(guidance_scale=1.0),
        "spherical": lambda: SphericalFlowSampler(),
        "blur": lambda: BlurAncestralSampler(),
        "blur_bootstrap": lambda: BlurAncestralSampler(
            self_conditioning_mode="bootstrap"
        ),
    }[factory]()
    for budget in (16, 17, 128):
        steps = sampler_steps(OmegaConf.create({}), sampler, budget)
        assert used(steps) <= budget < used(steps + 1)
