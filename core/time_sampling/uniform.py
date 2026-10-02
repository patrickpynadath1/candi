import torch


class UniformTimeSampler:
    def __init__(self, epsilon=1e-3, antithetic=True):
        if not 0 < epsilon < 1:
            raise ValueError("training time epsilon must be in (0, 1)")
        self.epsilon = epsilon
        self.antithetic = antithetic

    def sample(
        self,
        count,
        device,
        *,
        global_batch_size=None,
        world_size=1,
        rank=0,
        accumulation_steps=1,
        accumulation_index=0,
    ):
        n = count if global_batch_size is None else global_batch_size
        if n <= 0 or world_size <= 0 or accumulation_steps <= 0:
            raise ValueError("batch and partition sizes must be positive")
        if (
            not 0 <= rank < world_size
            or not 0 <= accumulation_index < accumulation_steps
        ):
            raise ValueError("invalid time sampling partition")
        values = torch.rand(n, device=device)
        if self.antithetic:
            values = (values / n + torch.arange(n, device=device) / n) % 1
        times = (1 - self.epsilon) * values + self.epsilon
        if global_batch_size is not None:
            partitions = world_size * accumulation_steps
            if n % partitions:
                raise ValueError(
                    "global batch must divide evenly across time partitions"
                )
            times = times.reshape(world_size, accumulation_steps, -1)[
                rank, accumulation_index
            ]
            if count > len(times):
                raise ValueError("local batch exceeds its global time partition")
            times = times[:count]
        return times
