"""Optimizer-step learning-rate schedules, independent of diffusion schedules."""

import math
from torch.optim.lr_scheduler import LambdaLR


def constant(optimizer):
    return LambdaLR(optimizer, lambda step: 1.0)


def constant_warmup(optimizer, warmup_steps=2500):
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    return LambdaLR(
        optimizer,
        lambda step: 1.0 if warmup_steps == 0 else min(1.0, step / warmup_steps),
    )


def step_scheduler(optimizer, warmup_steps=500, n_halve_steps=10000):
    """Match the original LRHalveScheduler, including its warmup boundary."""
    if warmup_steps < 0 or n_halve_steps <= 0:
        raise ValueError("require warmup_steps >= 0 and n_halve_steps > 0")

    def factor(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 ** ((step - warmup_steps) // n_halve_steps)

    return LambdaLR(optimizer, factor)


def cosine_decay_warmup(
    optimizer,
    total_steps,
    warmup_fraction=0.1,
    warmup_steps=None,
    warmup_lr_init=1e-6,
    lr_min=1e-6,
):
    """Linear warmup followed by one cosine decay, with absolute LR endpoints.

    Each parameter group keeps its own peak LR. The floor is held after the
    configured horizon; stepping a resumed run never restarts the cosine.
    """
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    warmup = (
        int(total_steps * warmup_fraction) if warmup_steps is None else warmup_steps
    )
    if total_steps <= 0 or not 0 <= warmup < total_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")
    factors = []
    for group in optimizer.param_groups:
        peak = group["lr"]
        if peak <= 0 or not 0 <= lr_min <= peak or not 0 <= warmup_lr_init <= peak:
            raise ValueError(
                "cosine endpoints must be between zero and each group's positive LR"
            )

        def factor(step, peak=peak):
            if step < warmup:
                return (warmup_lr_init + (peak - warmup_lr_init) * step / warmup) / peak
            progress = min(1.0, (step - warmup) / (total_steps - warmup))
            return (
                lr_min + (peak - lr_min) * (1 + math.cos(math.pi * progress)) / 2
            ) / peak

        factors.append(factor)
    return LambdaLR(optimizer, factors)
