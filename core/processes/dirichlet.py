"""Dirichlet corruption and bridges with configurable FP32/FP64 log states."""

import torch
from torch import nn
from torch.nn import functional as F
from .simplex_math import corruption_parameters, log_dirichlet_sample, reverse_bridge


class DirichletProcess(nn.Module):
    def corrupt_state(self, clean, coefficients, corruption_mask):
        return self.corrupt(clean, coefficients, corruption_mask)

    def prior_state(self, clean, coefficients):
        return self.prior(len(clean), clean.shape[1], coefficients)

    def __init__(self, vocab_size, reference=None, precision="float64"):
        super().__init__()
        if precision not in {"float32", "float64"}:
            raise ValueError("Dirichlet precision must be float32 or float64")
        dtype = getattr(torch, precision)
        prior = (
            torch.ones(vocab_size, dtype=dtype)
            if reference is None
            else torch.as_tensor(reference, dtype=dtype)
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
        return F.one_hot(tokens, self.vocab_size).to(self.reference).log()

    def corrupt(self, tokens, path, corruption_mask=None):
        clean = F.one_hot(tokens, self.vocab_size).to(self.reference)
        state = clean.log()
        positive = path.time > 0
        if positive.any():
            beta = corruption_parameters(
                clean[positive],
                path.time[positive].to(self.reference),
                path.concentration[positive].to(self.reference),
                self.reference,
            )
            state[positive] = log_dirichlet_sample(beta)
        if corruption_mask is not None:
            state = torch.where(corruption_mask.bool()[..., None], state, clean.log())
        return state

    def prior(self, batch_size, length, path):
        beta = path.concentration.to(self.reference)[:, None, None] * self.reference
        return log_dirichlet_sample(beta.expand(batch_size, length, self.vocab_size))

    def reverse(self, state, clean_tokens, destination, source, churn):
        return reverse_bridge(
            state.to(self.reference),
            clean_tokens,
            destination.time.to(self.reference),
            source.time.to(self.reference),
            destination.concentration.to(self.reference),
            source.concentration.to(self.reference),
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
