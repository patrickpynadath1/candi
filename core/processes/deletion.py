"""Forward deletion, with alignment available only to training targets."""

from dataclasses import dataclass
import torch
from core.sequence_edits import SequenceEdit, valid_positions


@dataclass(frozen=True)
class DeletionSample:
    edit: SequenceEdit
    birth_times: torch.Tensor
    deleted: torch.Tensor
    gap_for_token: torch.Tensor
    gap_counts: torch.Tensor


class DeletionProcess:
    def corrupt(self, valid, editable, time, schedule):
        if valid.shape != editable.shape or (editable & ~valid).any():
            raise ValueError("editable positions must be valid sequence positions")
        if not torch.equal(valid, valid_positions(valid.sum(1), valid.shape[1])):
            raise ValueError("valid sequence positions must precede right padding")
        births = schedule.birth_times(valid.shape, valid.device)
        keep = valid & (~editable | (births >= time[:, None]))
        edit = SequenceEdit.deletion(keep)
        deleted = valid & ~keep
        gaps = keep.long().cumsum(1) - keep.long()
        counts = torch.zeros(
            (len(valid), edit.source.shape[1] + 1),
            device=valid.device,
            dtype=torch.long,
        )
        counts.scatter_add_(1, gaps.clamp_max(counts.shape[1] - 1), deleted.long())
        return DeletionSample(edit, edit.apply(births), deleted, gaps, counts)
