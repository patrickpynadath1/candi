"""Composable tensor operations for posterior parameterizations."""

import math
import torch


def validate_softcap(softcap):
    if softcap is not None and (not math.isfinite(softcap) or softcap <= 0):
        raise ValueError("softcap must be finite and positive, or None")


def softcap_logits(logits, softcap):
    """Bound raw model logits before constraints or inference-only filtering."""
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)


def exclude_token(logits, token_id):
    values = logits.clone()
    values[..., token_id] = -torch.inf
    return values


def scale_temperature(logits, temperature):
    return logits / temperature


def filter_top_k(logits, k):
    if k == 0:
        return logits
    threshold = logits.topk(min(k, logits.shape[-1]), -1).values[..., -1:]
    return logits.masked_fill(logits < threshold, -torch.inf)


def filter_top_p(logits, p):
    if p == 1:
        return logits
    values, indices = logits.sort(descending=True, dim=-1)
    probabilities = values.softmax(-1)
    remove = probabilities.cumsum(-1) - probabilities >= p
    return logits.masked_fill(
        torch.zeros_like(remove).scatter(-1, indices, remove), -torch.inf
    )


def greedy_projection(values):
    chosen = values.argmax(-1, keepdim=True)
    return torch.full_like(values, -torch.inf).scatter(-1, chosen, 0)


def greedy_probabilities(log_probs):
    """Project a prediction to a one-hot distribution in its original dtype."""
    return greedy_projection(log_probs).exp()


def normalize_log_probs(values):
    # Preserve the baseline's explicit logsumexp operation and input precision.
    return values - torch.logsumexp(values, dim=-1, keepdim=True)


def fix_observed_tokens(log_probs, observed, mask_index):
    fixed = torch.full_like(log_probs, -torch.inf).scatter(-1, observed[..., None], 0)
    return torch.where(observed.ne(mask_index)[..., None], fixed, log_probs)


def predictive_entropy(log_probs):
    probabilities = log_probs.float().exp()
    return -torch.special.xlogy(probabilities, probabilities).sum(-1)
