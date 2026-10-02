"""Construct Lightning strategies and guard unsupported sharded lifecycles."""

from hydra.utils import instantiate
from lightning.pytorch.strategies import FSDPStrategy


def build_strategy(config, algorithm):
    selected = config.get("strategy")
    strategy = (
        instantiate(selected) if selected is not None else config.trainer.strategy
    )
    if strategy == "ddp":
        return "ddp_find_unused_parameters_true"
    if isinstance(strategy, FSDPStrategy) or (
        isinstance(strategy, str) and strategy.startswith("fsdp")
    ):
        if config.training.ema > 0:
            raise ValueError(
                "FSDP currently requires training.ema=0; EMA does not track sharded parameters"
            )
        callback = config.get("callback")
        if callback is not None and callback.get("every_n_steps", 0):
            raise ValueError(
                "FSDP requires callback.every_n_steps=0; use standalone evaluation of full checkpoints"
            )
        if config.trainer.get("gradient_clip_val", 0):
            raise ValueError(
                "FSDP currently requires trainer.gradient_clip_val=0; norm clipping is unsupported by the installed FSDP precision plugin"
            )
        if any(
            callable(getattr(module, "post_optimizer_step", None))
            for module in algorithm.modules()
        ):
            raise ValueError(
                "FSDP does not yet support post_optimizer_step parameter projections; use DDP for this algorithm"
            )
    return strategy
