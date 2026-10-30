"""Training block layouts sampled independently of corruption and generation."""

import math
import torch
from torch import nn


class BlockSizeMixture(nn.Module):
    def __init__(self, sizes, probabilities, sampling="random", seed=0, overrides=None):
        super().__init__()
        if not sizes or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("block sizes must be positive integers")
        if len(set(sizes)) != len(sizes):
            raise ValueError("block sizes must be unique")
        if (
            len(probabilities) != len(sizes)
            or any(not math.isfinite(p) or p <= 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0)
        ):
            raise ValueError("positive block probabilities must sum to one")
        if sampling not in {"random", "stratified"}:
            raise ValueError("sampling must be random or stratified")
        self.sizes = tuple(sizes)
        self.cumulative = torch.tensor(probabilities, dtype=torch.float64).cumsum(0)
        self.sampling, self.seed = sampling, int(seed)
        self.overrides = {}
        self.objectives = nn.ModuleDict()
        self.time_samplers = nn.ModuleDict()
        for size, options in (overrides or {}).items():
            size = int(size)
            if size not in sizes:
                raise ValueError("override size must belong to the mixture")
            options = dict(options)
            if set(options) - {"time_sampler", "objective", "pure_noise"}:
                raise ValueError("unsupported block training override")
            objective = options.pop("objective", None)
            if objective is not None:
                self.objectives[str(size)] = objective
            time_sampler = options.get("time_sampler")
            if isinstance(time_sampler, nn.Module):
                self.time_samplers[str(size)] = options.pop("time_sampler")
            self.overrides[size] = options

    def sample(
        self,
        *,
        step=0,
        world_size=1,
        rank=0,
        accumulation_steps=1,
        accumulation_index=0,
        **unused,
    ):
        if (
            world_size < 1
            or accumulation_steps < 1
            or not 0 <= rank < world_size
            or not 0 <= accumulation_index < accumulation_steps
        ):
            raise ValueError("invalid block sampling partition")
        if len(self.sizes) == 1:
            return self.sizes[0]
        if self.sampling == "stratified":
            # Shared deterministic draw across ranks and accumulation slots, also
            # reproducible after checkpoint resume without mutable RNG state.
            generator = torch.Generator().manual_seed(self.seed + int(step))
            slots = world_size * accumulation_steps
            values = (
                torch.arange(slots, dtype=torch.float64) / slots
                + torch.rand((), generator=generator)
            ) % 1
            values = values[torch.randperm(slots, generator=generator)]
            value = values[accumulation_index * world_size + rank]
        else:
            value = torch.rand(())
        index = torch.searchsorted(self.cumulative, value, right=True).item()
        return self.sizes[min(index, len(self.sizes) - 1)]

    def loss_options(self, size):
        options = dict(self.overrides.get(size, {}))
        if str(size) in self.objectives:
            options["objective"] = self.objectives[str(size)]
        if str(size) in self.time_samplers:
            options["time_sampler"] = self.time_samplers[str(size)]
        return options
