from __future__ import annotations

import torch


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.float()
    return -torch.special.xlogy(probs, probs).sum(dim=-1)


def entropy_prefix_size(entropy: torch.Tensor, gamma: float) -> int:
    if entropy.numel() == 0:
        return 0
    accumulated = entropy.cumsum(dim=0)
    largest = entropy.cummax(dim=0).values
    tolerance = torch.finfo(entropy.dtype).eps * max(1.0, abs(gamma)) * 8
    return int(((accumulated - largest) <= gamma + tolerance).sum().item())


def select_entropy_bounded(entropy, active, gamma):
    """Select the bounded lowest-entropy prefix independently per example."""
    selected = torch.zeros_like(active, dtype=torch.bool)
    counts = []
    for batch_index in range(active.shape[0]):
        ids = active[batch_index].nonzero(as_tuple=False).squeeze(-1)
        ordered = ids[entropy[batch_index, ids].argsort()]
        k = entropy_prefix_size(entropy[batch_index, ordered], gamma)
        selected[batch_index, ordered[:k]] = True
        counts.append(k)
    return selected, counts


def sample_probabilities(probs):
    return torch.multinomial(
        probs.float().reshape(-1, probs.shape[-1]), 1
    ).reshape(probs.shape[:-1])
