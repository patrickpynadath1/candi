"""Uniform replacement corruption and plug-in reverse transitions."""

import torch
from torch.nn import functional as F
from core.samplers.utils import sample_categorical


class UniformProcess:
    prediction_kind = "plug_in"

    def __init__(self, vocab_size):
        if vocab_size < 2:
            raise ValueError("uniform process needs at least two tokens")
        self.vocab_size = vocab_size

    def prior(self, shape, device):
        return torch.randint(0, self.vocab_size, shape, device=device)

    def corrupt(self, tokens, keep, corruption_mask):
        moves = torch.rand(tokens.shape, device=tokens.device) < 1 - keep[:, None]
        replacements = torch.randint(
            0, self.vocab_size, tokens.shape, device=tokens.device
        )
        state = torch.where(moves, replacements, tokens)
        return torch.where(corruption_mask.bool(), state, tokens)

    def commit(self, estimate, deterministic=False):
        if deterministic:
            return estimate.argmax(-1)
        return torch.multinomial(
            estimate.float().reshape(-1, self.vocab_size), 1
        ).reshape(estimate.shape[:-1])

    def reverse_probabilities(
        self,
        state,
        estimate,
        keep_t,
        keep_s,
    ):

        if torch.any((keep_t <= 0) | (keep_s > 1) | (keep_s < keep_t)):
            raise ValueError("require 0 < keep_t <= keep_s <= 1")
        at, ass = keep_t[:, None, None], keep_s[:, None, None]
        ratio = at / ass
        observed = F.one_hot(state, self.vocab_size).to(estimate)
        return (
            at * self.vocab_size * estimate * observed
            + (ratio - at) * observed
            + (ass - at) * estimate
            + (1 - ratio) * (1 - ass) / self.vocab_size
        ) / (at * self.vocab_size * estimate.gather(-1, state[..., None]) + (1 - at))

    def reverse(
        self,
        state,
        estimate,
        keep_t,
        keep_s,
    ):
        return sample_categorical(
            self.reverse_probabilities(state, estimate, keep_t, keep_s)
        )
