"""
Spatially-blurred anchor means for the neighbor-mean r_a parameterization in SJD.

Replaces the SJD r_a center alpha(t) * e(a_i) with alpha(t) * mu_bar_i, where
mu_bar_i is a Gaussian-weighted average of neighboring anchor embeddings.
"""

import torch
import torch.nn as nn


def gaussian_position_kernel(
    seq_len: int,
    sigma: float,
    include_self: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build an (L, L) row-stochastic matrix W where W[i, j] is the Gaussian
    weight of position j when blurring at position i.

    W[i, j] = exp(-(i - j)^2 / (2 sigma^2))   (with W[i, i] zeroed if include_self=False)

    Then each row is normalized to sum to 1, which automatically handles boundary
    truncation (rows near the edges sum the same as interior rows post-normalization).

    Args:
        seq_len: sequence length L.
        sigma: spatial Gaussian std in positions. sigma -> 0 recovers identity
            (assuming include_self=True). Larger sigma = more blurring.
        include_self: if False, W[i, i] = 0 (blur uses neighbors only).
        device, dtype: torch tensor options.

    Returns:
        W: (L, L) tensor, rows sum to 1.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    idx = torch.arange(seq_len, device=device, dtype=dtype)
    # (L, L) matrix of squared position differences
    diff_sq = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2  # [i, j] = (i - j)^2
    logits = -diff_sq / (2.0 * sigma * sigma)

    if not include_self:
        # Mask out the diagonal before softmax. Use -inf so it gets zero mass.
        logits = logits.masked_fill(
            torch.eye(seq_len, dtype=torch.bool, device=device), float("-inf")
        )

    W = torch.softmax(logits, dim=-1)  # row-normalize
    
    W_norm = W / W.diag().unsqueeze(-1)             # row i scaled so W[i, i] = 1

    return W_norm


def neighbor_blurred_means(
    orig_means: nn.Embedding | torch.Tensor,
    kernel: torch.Tensor | None = None,
) -> torch.Tensor:

    # mu_bar[b, i, :] = sum_j W[i, j] * E[b, j, :]
    # Equivalently: mu_bar = W @ E (broadcast over batch).
    if kernel.device != orig_means.device:
        kernel = kernel.to(orig_means.device)

    if kernel.dtype != orig_means.dtype:
        kernel = kernel.to(orig_means.dtype)

    mu_bar = torch.einsum("ij,bjd->bid", kernel, orig_means)
    return mu_bar


def r_a_means(
    anchor_ids: torch.Tensor,
    embedding: nn.Embedding | torch.Tensor,
    alpha_t: float | torch.Tensor,
    sigma_spatial: float,
    include_self: bool = True,
    kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convenience wrapper: returns the full r_a center alpha(t) * mu_bar_i.

    Args:
        anchor_ids: (B, L) token IDs.
        embedding: nn.Embedding or (V, d) tensor.
        alpha_t: scalar or (B,) or (B, 1, 1) — the VP schedule's alpha(t).
        sigma_spatial: spatial Gaussian std.
        include_self, kernel: see neighbor_blurred_means.

    Returns:
        (B, L, d) tensor of r_a means.
    """
    mu_bar = neighbor_blurred_means(
        anchor_ids, embedding, sigma_spatial, include_self=include_self, kernel=kernel
    )

    if isinstance(alpha_t, torch.Tensor):
        # Reshape alpha_t to broadcast over (B, L, d).
        while alpha_t.dim() < mu_bar.dim():
            alpha_t = alpha_t.unsqueeze(-1)

    return alpha_t * mu_bar


# ---------- Smoke test ----------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, L, V, d = 2, 8, 100, 16
    embed = nn.Embedding(V, d)
    ids = torch.randint(0, V, (B, L))

    # Sanity: sigma -> 0 with self-inclusion should recover identity.
    W_small = gaussian_position_kernel(L, sigma=1e-3, include_self=True)
    assert torch.allclose(W_small, torch.eye(L), atol=1e-3), "sigma->0 should be identity"
    mu_id = neighbor_blurred_means(ids, embed, sigma=1e-3)
    mu_raw = embed(ids)
    print("sigma->0 matches raw embeddings:", torch.allclose(mu_id, mu_raw, atol=1e-3))

    # Larger sigma blurs.
    W = gaussian_position_kernel(L, sigma=1.5)
    print("\nKernel rows sum to 1:", torch.allclose(W.sum(-1), torch.ones(L)))
    print("W[4]:", W[4].numpy().round(3))

    # Without self.
    W_ns = gaussian_position_kernel(L, sigma=1.5, include_self=False)
    print("\nNo-self diag is zero:", (W_ns.diag() == 0).all().item())
    print("W_ns[4]:", W_ns[4].numpy().round(3))

    # Full pipeline.
    alpha_t = torch.tensor([0.7, 0.3])
    means = r_a_means(ids, embed, alpha_t, sigma_spatial=1.5)
    print("\nr_a_means shape:", means.shape)  # (B, L, d)

    # Precomputed kernel path.
    kernel = gaussian_position_kernel(L, sigma=1.5, device=ids.device, dtype=torch.float32)
    means2 = r_a_means(ids, embed, alpha_t, sigma_spatial=1.5, kernel=kernel)
    print("Precomputed kernel matches:", torch.allclose(means, means2, atol=1e-6))