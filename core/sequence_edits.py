"""Representation-independent, batched sequence edits.

Padding is storage, not a token. An edit maps surviving positions and new gaps
once, then applies that same map to every view and any trajectory feedback.
"""

from dataclasses import dataclass
import torch


def valid_positions(lengths, width=None):
    width = int(lengths.max()) if width is None else width
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


def gather_positions(value, indices):
    """Gather sequence positions, including empty input sequences."""
    shape = (*indices.shape, *value.shape[2:])
    if value.shape[1] == 0:
        return value.new_zeros(shape)
    index = indices.clamp_min(0)
    for _ in value.shape[2:]:
        index = index.unsqueeze(-1)
    return value.gather(1, index.expand(shape))


@dataclass(frozen=True)
class SequenceEdit:
    source: torch.Tensor  # old position, or -1 for insertion/padding
    gap: torch.Tensor  # old gap for births; -1 otherwise
    lengths: torch.Tensor

    @property
    def valid(self):
        return valid_positions(self.lengths, self.source.shape[1])

    @property
    def inserted(self):
        return self.valid & (self.gap >= 0)

    def apply(self, old, births=None, *, padding=0):
        """Birth values are indexed by OLD gap, not by shifted token position."""
        result = gather_positions(old, self.source)
        valid, inserted = self.valid, self.inserted
        if births is not None:
            new = gather_positions(births, self.gap)
            selector = inserted.reshape(*inserted.shape, *([1] * (old.ndim - 2)))
            result = torch.where(selector, new, result)
        elif inserted.any():
            selector = inserted.reshape(*inserted.shape, *([1] * (old.ndim - 2)))
            result = torch.where(selector, old.new_tensor(padding), result)
        valid = valid.reshape(*valid.shape, *([1] * (old.ndim - 2)))
        return torch.where(valid, result, old.new_tensor(padding))

    @classmethod
    def deletion(cls, keep):
        if keep.ndim != 2 or keep.dtype != torch.bool:
            raise ValueError("keep must be boolean [batch, length]")
        lengths = keep.sum(1)
        width = int(lengths.max())
        positions = torch.arange(keep.shape[1], device=keep.device).expand_as(keep)
        source = positions.masked_fill(~keep, keep.shape[1]).sort(1).values[:, :width]
        source = source.masked_fill(~valid_positions(lengths, width), -1)
        return cls(source, torch.full_like(source, -1), lengths)

    @classmethod
    def insertion(cls, lengths, counts):
        if counts.ndim != 2 or counts.shape[0] != len(lengths):
            raise ValueError("insertion counts must be [batch, old_width+1]")
        if counts.dtype != torch.long or (counts < 0).any():
            raise ValueError("insertion counts must be nonnegative integers")
        if (lengths < 0).any() or (lengths >= counts.shape[1]).any():
            raise ValueError("lengths exceed insertion gap storage")
        gaps_valid = valid_positions(lengths + 1, counts.shape[1])
        if counts.masked_select(~gaps_valid).any():
            raise ValueError("cannot insert into padding gaps")
        new_lengths = lengths + counts.sum(1)
        width = int(new_lengths.max())
        source = counts.new_full((len(lengths), width), -1)
        gap = source.clone()
        # Structural indexing is intentionally separate from differentiable state
        # movement. Vectorize the per-row expansion; never loop over insertions.
        for row, length in enumerate(lengths.tolist()):
            count = counts[row, : length + 1]
            repeats = count + 1
            owners = torch.repeat_interleave(
                torch.arange(length + 1, device=counts.device), repeats
            )
            ends = repeats.cumsum(0) - 1
            is_old = torch.zeros(len(owners), dtype=torch.bool, device=counts.device)
            is_old[ends[:-1]] = True
            # The last repeated element is a virtual right boundary, not a token.
            owners, is_old = owners[:-1], is_old[:-1]
            n = len(owners)
            source[row, :n] = torch.where(is_old, owners, -1)
            gap[row, :n] = torch.where(is_old, -1, owners)
        return cls(source, gap, new_lengths)
