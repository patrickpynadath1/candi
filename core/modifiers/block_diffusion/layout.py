"""Two-stream positions and visibility for block diffusion."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class BlockLayout:
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    block_ids: torch.Tensor


def make_layout(length, block_size, device, *, batch_size=1, offset=0,
                clean_length=None, x0_causal=True):
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    clean_length = length if clean_length is None else clean_length
    positions = torch.arange(offset, offset + length, device=device)
    clean_positions = torch.arange(clean_length, device=device)
    blocks = positions.div(block_size, rounding_mode="floor")
    clean_blocks = clean_positions.div(block_size, rounding_mode="floor")
    clean_q, clean_k = clean_positions[:, None], clean_positions[None, :]
    clean_to_clean = (
        clean_k <= clean_q if x0_causal
        else clean_blocks[None, :] <= clean_blocks[:, None]
    )
    noisy_to_clean = clean_blocks[None, :] < blocks[:, None]
    noisy_to_noisy = blocks[None, :] == blocks[:, None]
    visible = torch.cat(
        [
            torch.cat([clean_to_clean, torch.zeros(clean_length, length, dtype=torch.bool, device=device)], 1),
            torch.cat([noisy_to_clean, noisy_to_noisy], 1),
        ],
        0,
    )
    # SDPA boolean masks use True for allowed attention.
    mask = visible[None, None].expand(batch_size, 1, -1, -1)
    ids = torch.cat([clean_positions, positions])[None].expand(batch_size, -1)
    return BlockLayout(mask, ids, blocks)
