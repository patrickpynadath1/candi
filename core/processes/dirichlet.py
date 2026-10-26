"""Dirichlet corruption and bridges; opt-in precision and row chunking."""

import torch
from torch import nn
from torch.nn import functional as F
from .simplex_math import corruption_parameters, log_dirichlet_sample, reverse_bridge


class DirichletProcess(nn.Module):
    def corrupt_state(self, clean, coefficients, corruption_mask):
        return self.corrupt(clean, coefficients, corruption_mask)

    def prior_state(self, clean, coefficients):
        return self.prior(len(clean), clean.shape[1], coefficients)

    def __init__(
        self,
        vocab_size,
        reference=None,
        precision="float64",
        sampler_backend="torch",
        chunk_size=0,
    ):
        super().__init__()
        precision = {"float32": "fp32", "float64": "fp64"}.get(precision, precision)
        if precision not in {"fp32", "fp64"} or sampler_backend not in {
            "torch",
            "triton",
        }:
            raise ValueError("invalid simplex precision or sampler backend")
        if chunk_size < 0:
            raise ValueError("chunk_size must be nonnegative")
        dtype = torch.float32 if precision == "fp32" else torch.float64
        self.sampler_backend, self.chunk_size = sampler_backend, int(chunk_size)
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

    @property
    def state_dtype(self):
        return self.reference.dtype

    def encode(self, tokens):
        return F.one_hot(tokens, self.vocab_size).to(self.state_dtype).log()

    def corrupt(self, tokens, path, corruption_mask=None):
        if self.chunk_size:
            return self._corrupt_chunked(tokens, path, corruption_mask)
        clean = F.one_hot(tokens, self.vocab_size).to(self.state_dtype)
        state = clean.log()
        positive = path.time > 0
        if positive.any():
            beta = corruption_parameters(
                clean[positive],
                path.time[positive].to(self.state_dtype),
                path.concentration[positive].to(self.state_dtype),
                self.reference,
            )
            state[positive] = log_dirichlet_sample(beta, self.sampler_backend)
        if corruption_mask is not None:
            state = torch.where(corruption_mask.bool()[..., None], state, clean.log())
        return state

    def _corrupt_chunked(self, tokens, path, corruption_mask):
        # Row chunks include the entire vocabulary: no truncation or separate
        # vocabulary normalization. CopySlices retains concentration gradients.
        flat = tokens.reshape(-1)
        out = torch.empty(
            (len(flat), self.vocab_size), device=tokens.device, dtype=self.state_dtype
        )
        mask = None if corruption_mask is None else corruption_mask.reshape(-1).bool()
        for start in range(0, len(flat), self.chunk_size):
            end = min(start + self.chunk_size, len(flat))
            rows = torch.arange(start, end, device=tokens.device) // tokens.shape[1]
            t = path.time[rows].to(self.state_dtype)[:, None]
            c = path.concentration[rows].to(self.state_dtype)[:, None]
            clean = torch.zeros(
                (end - start, self.vocab_size),
                device=tokens.device,
                dtype=self.state_dtype,
            )
            clean.scatter_(1, flat[start:end, None], 1)
            beta = c * ((1 - t) * clean + t * self.reference)
            sampled = log_dirichlet_sample(
                torch.where(t > 0, beta, clean), self.sampler_backend
            )
            active = t > 0
            if mask is not None:
                active = active & mask[start:end, None]
            out[start:end] = torch.where(active, sampled, clean.log())
        return out.reshape(*tokens.shape, self.vocab_size)

    def prior(self, batch_size, length, path):
        if self.chunk_size:
            tokens = torch.zeros(
                (batch_size, length), device=self.reference.device, dtype=torch.long
            )
            from types import SimpleNamespace

            return self._corrupt_chunked(
                tokens,
                SimpleNamespace(
                    time=torch.ones_like(path.time), concentration=path.concentration
                ),
                None,
            )
        beta = path.concentration.to(self.state_dtype)[:, None, None] * self.reference
        return log_dirichlet_sample(
            beta.expand(batch_size, length, self.vocab_size), self.sampler_backend
        )

    def reverse(self, state, clean_tokens, destination, source, churn):
        return reverse_bridge(
            state.to(self.reference),
            clean_tokens,
            destination.time.to(self.state_dtype),
            source.time.to(self.state_dtype),
            destination.concentration.to(self.state_dtype),
            source.concentration.to(self.state_dtype),
            self.reference,
            churn,
            self.sampler_backend,
        )

    def drop_condition(self, state, corruption_mask, probability):
        if not probability or corruption_mask is None:
            return state
        drop = torch.rand(len(state), device=state.device) < probability
        return state.masked_fill(
            drop[:, None, None] & ~corruption_mask.bool()[..., None], -torch.inf
        )
