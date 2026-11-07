"""Allocate a sequence-level inference budget to the inner sampler."""

from math import ceil
from omegaconf import OmegaConf


def sampler_steps(config, sampler, nfe):
    """Reserve equal NFE for every block, conservatively including prompt blocks."""
    blocks = 1
    if OmegaConf.select(config, "block_diffusion.enabled", default=False):
        if OmegaConf.select(config, "task.variable_length", default=False):
            raise ValueError("A fixed sequence NFE budget requires fixed-length blocks")
        blocks = ceil(config.task.length / config.block_diffusion.block_size)
    elif OmegaConf.select(config, "algorithm._target_", default="").endswith(
        "LatentAutoencoderDiffusion"
    ):
        # The latent sampler spends its budget once per latent block.
        blocks = ceil(config.task.length / config.latent.block_size)
    per_block = nfe // blocks
    if per_block < 2:
        raise ValueError("NFE budget must allow at least two evaluations per block")
    return getattr(sampler, "steps_for_nfe", lambda n: n - 1)(per_block)
