"""Cross-position blending kernels for spatially smoothed CANDI."""

import torch


def gaussian_position_kernel(
    seq_len: int,
    bandwidth: float,
    *,
    include_self: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the row-stochastic 1D Gaussian position kernel from SJD.

    A zero bandwidth is the identity-kernel CANDI limit. At sequence boundaries
    the truncated Gaussian is renormalized so every row still sums to one.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if bandwidth < 0:
        raise ValueError(f"bandwidth must be non-negative, got {bandwidth}")
    if bandwidth == 0:
        if not include_self:
            raise ValueError("bandwidth=0 requires include_self=true")
        return torch.eye(seq_len, device=device, dtype=dtype)
    if seq_len == 1 and not include_self:
        raise ValueError("cannot exclude self from a length-one kernel")

    positions = torch.arange(seq_len, device=device, dtype=dtype)
    squared_distance = (positions[:, None] - positions[None, :]).square()
    logits = -squared_distance / (2 * bandwidth**2)
    if not include_self:
        logits.fill_diagonal_(float("-inf"))
    return torch.softmax(logits, dim=-1)


def blend_positions(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply an ``(L, L)`` position kernel to ``(B, L, D)`` values."""
    if values.ndim != 3:
        raise ValueError(f"values must have shape (B, L, D), got {values.shape}")
    if kernel.shape != (values.shape[1], values.shape[1]):
        raise ValueError(
            f"kernel shape {kernel.shape} does not match sequence length "
            f"{values.shape[1]}"
        )
    return torch.einsum(
        "ij,bjd->bid", kernel.to(device=values.device, dtype=values.dtype), values
    )
