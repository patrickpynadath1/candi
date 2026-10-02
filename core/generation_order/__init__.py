"""Generation order policies independent of model and sampling execution."""

import torch


class GenerationOrder:
    def __init__(self, mode="random", block_size=None, ar_probability=0.0):
        if mode not in {"sequential", "random", "block"}:
            raise ValueError("order mode must be sequential, random or block")
        if mode == "block" and (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size <= 0
        ):
            raise ValueError("block order requires a positive block_size")
        if not 0 <= ar_probability <= 1:
            raise ValueError("ar_probability must be in [0, 1]")
        self.mode, self.block_size, self.ar_probability = (
            mode,
            block_size,
            ar_probability,
        )

    def sample(self, batch_size, length, *, device):
        sequential = torch.arange(length, device=device).expand(batch_size, -1)
        if self.mode == "sequential" or length == 0:
            return sequential
        if self.mode == "random":
            order = torch.rand(batch_size, length, device=device).argsort(-1)
        else:
            order = torch.cat(
                [
                    torch.rand(
                        batch_size, min(self.block_size, length - start), device=device
                    ).argsort(-1)
                    + start
                    for start in range(0, length, self.block_size)
                ],
                -1,
            )
        if self.ar_probability:
            use_ar = torch.rand(batch_size, device=device) < self.ar_probability
            order = torch.where(use_ar[:, None], sequential, order)
        return order

    @staticmethod
    def observed_first(order, observed):
        """Stable partition: observed positions first, retaining chosen order."""
        return order.gather(
            1, (~observed.gather(1, order)).int().argsort(dim=-1, stable=True)
        )
