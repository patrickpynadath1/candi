"""Generic independent-span topology for sequence-valued processes."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SequenceLayout:
    """Cuts that prevent sequence processes from coupling positions."""

    boundaries: torch.Tensor

    def __post_init__(self):
        if self.boundaries.ndim != 2 or self.boundaries.dtype != torch.bool:
            raise ValueError("sequence boundaries must be boolean [batch, length+1]")
        if not self.boundaries[:, 0].all() or not self.boundaries[:, -1].all():
            raise ValueError("sequence layouts must include outer boundaries")

    @classmethod
    def contiguous(cls, batch_size, length, device):
        value = torch.zeros(batch_size, length + 1, dtype=torch.bool, device=device)
        value[:, 0] = value[:, -1] = True
        return cls(value)

    @classmethod
    def fixed_blocks(cls, batch_size, length, block_size, device, *, offset=0):
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        value = cls.contiguous(batch_size, length, device).boundaries.clone()
        positions = torch.arange(length + 1, device=device) + offset
        value[:, (positions % block_size) == 0] = True
        return cls(value)

    def spans(self, mask):
        """Yield effective ``(batch, start, end)`` independent true runs."""
        if mask.shape != (self.boundaries.shape[0], self.boundaries.shape[1] - 1):
            raise ValueError("sequence mask and layout shapes do not match")
        mask = mask.bool()
        starts = mask & (
            self.boundaries[:, :-1] | ~torch.nn.functional.pad(mask[:, :-1], (1, 0))
        )
        ends = mask & (
            self.boundaries[:, 1:] | ~torch.nn.functional.pad(mask[:, 1:], (0, 1))
        )
        start_rows = starts.nonzero(as_tuple=False).cpu().tolist()
        end_rows = ends.nonzero(as_tuple=False).cpu().tolist()
        if len(start_rows) != len(end_rows):
            raise RuntimeError("invalid effective sequence span topology")
        for (batch, start), (end_batch, end) in zip(start_rows, end_rows):
            if batch != end_batch:
                raise RuntimeError("invalid effective sequence span ordering")
            yield batch, start, end + 1


def resolve_sequence_layout(mask, layout=None):
    if layout is None:
        return SequenceLayout.contiguous(*mask.shape, mask.device)
    if not isinstance(layout, SequenceLayout):
        raise TypeError("sequence_layout must be a SequenceLayout")
    return layout
