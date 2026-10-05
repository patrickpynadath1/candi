"""Dirichlet corruption and bridges with explicit inputs and FP64 log states."""

import torch
from torch import nn
from torch.nn import functional as F
from .simplex_math import corruption_parameters, log_dirichlet_sample, reverse_bridge


class DirichletProcess(nn.Module):
    def corrupt_state(self, clean, coefficients, corruption_mask):
        return self.corrupt(clean, coefficients, corruption_mask)

    def prior_state(self, clean, coefficients):
        return self.prior(len(clean), clean.shape[1], coefficients)

    def __init__(self, vocab_size, reference=None):
        super().__init__()
        prior = (
            torch.ones(vocab_size, dtype=torch.float64)
            if reference is None
            else torch.as_tensor(reference, dtype=torch.float64)
        )
        if (
            prior.shape != (vocab_size,)
            or not torch.isfinite(prior).all()
            or (prior < 0).any()
            or prior.sum() <= 0
        ):
            raise ValueError(
                "reference must contain one finite nonnegative weight per token"
            )
        self.register_buffer("reference", prior / prior.sum())
        self.vocab_size = vocab_size

    def encode(self, tokens):
        return F.one_hot(tokens, self.vocab_size).double().log()

    def corrupt(self, tokens, path, corruption_mask=None):
        clean = F.one_hot(tokens, self.vocab_size).double()
        state = clean.log()
        positive = path.time > 0
        if positive.any():
            beta = corruption_parameters(
                clean[positive],
                path.time[positive],
                path.concentration[positive],
                self.reference,
            )
            state[positive] = log_dirichlet_sample(beta)
        if corruption_mask is not None:
            state = torch.where(corruption_mask.bool()[..., None], state, clean.log())
        return state

    def prior(self, batch_size, length, path):
        beta = path.concentration[:, None, None] * self.reference
        return log_dirichlet_sample(beta.expand(batch_size, length, self.vocab_size))

    def reverse(self, state, clean_tokens, destination, source, churn):
        return reverse_bridge(
            state,
            clean_tokens,
            destination.time,
            source.time,
            destination.concentration,
            source.concentration,
            self.reference,
            churn,
        )

    def drop_condition(self, state, corruption_mask, probability):
        if not probability or corruption_mask is None:
            return state
        drop = torch.rand(len(state), device=state.device) < probability
        return state.masked_fill(
            drop[:, None, None] & ~corruption_mask.bool()[..., None], -torch.inf
        )
