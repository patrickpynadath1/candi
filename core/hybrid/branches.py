"""Branch protocols for parallel noisy views of the same clean tokens.

Each branch owns its local process, schedule, representation and clock.  The
coordinator never implements a process transition.
"""

from dataclasses import dataclass
import torch
from torch import nn

from .time_maps import IdentityTimeMap


@dataclass(frozen=True)
class BranchPoint:
    time: torch.Tensor
    coefficients: object


class _Branch(nn.Module):
    def __init__(self, process, schedule, adapter, time_map=None, name=None):
        super().__init__()
        self.process, self.schedule, self.adapter = process, schedule, adapter
        self.time_map = time_map or IdentityTimeMap()
        self.name = name or type(self).__name__

    def point(self, global_time):
        local = self.time_map(global_time)
        return BranchPoint(local, self.schedule(local))

    def embed(self, state, point):
        return self.adapter(state, point)

    def clean_embedding(self, probabilities):
        return self.adapter.clean_embedding(probabilities)


class DiscreteBranch(_Branch):
    """A discrete keep-rate process, including absorbing and uniform noise."""

    def corrupt(self, tokens, point, corruption_mask):
        keep = (
            point.coefficients[0]
            if isinstance(point.coefficients, tuple)
            else point.coefficients
        )
        return self.process.corrupt(tokens, keep, corruption_mask)

    def prior(self, tokens, point, prompt_mask):
        try:
            return self.process.prior(tokens.shape, tokens.device, tokens, prompt_mask)
        except TypeError:
            state = self.process.prior(tokens.shape, tokens.device)
            return torch.where(prompt_mask.bool(), tokens, state)

    def update(self, state, probabilities, source, destination, *, noise=None):
        del noise
        keep_t = (
            source.coefficients[0]
            if isinstance(source.coefficients, tuple)
            else source.coefficients
        )
        keep_s = (
            destination.coefficients[0]
            if isinstance(destination.coefficients, tuple)
            else destination.coefficients
        )
        return self.process.reverse(state, probabilities, keep_t, keep_s)

    def clamp_prompt(self, state, tokens, prompt_mask, point):
        del point
        return torch.where(prompt_mask.bool(), tokens, state)


class GaussianBranch(_Branch):
    def __init__(
        self, process, schedule, representation, adapter, time_map=None, name=None
    ):
        super().__init__(process, schedule, adapter, time_map, name)
        self.representation = representation

    def corrupt(self, tokens, point, corruption_mask):
        clean = self.representation.encode(tokens)
        path = point.coefficients
        return self.process.corrupt(clean, path.alpha, path.sigma, corruption_mask)

    def prior(self, tokens, point, prompt_mask):
        clean = self.representation.encode(tokens)
        state = self.process.prior(clean, point.coefficients.sigma)
        return self.clamp_prompt(state, tokens, prompt_mask, point)

    def update(self, state, probabilities, source, destination, *, noise=None):
        clean = self.representation.posterior_mean(probabilities)
        noise = torch.randn_like(state) if noise is None else noise
        return self.process.reverse_posterior(
            state, clean, source.coefficients, destination.coefficients, noise
        )

    def clamp_prompt(self, state, tokens, prompt_mask, point):
        clean = self.representation.encode(tokens)
        # Fixed clean conditioning is independent of the current noisy clock.
        return torch.where(prompt_mask.bool()[..., None], clean, state)

    def clamp_committed(self, state, tokens, committed):
        clean = self.representation.encode(
            tokens.clamp_max(self.representation.spec.vocab_size - 1)
        )
        return torch.where(committed[..., None], clean, state)


class SimplexBranch(_Branch):
    def corrupt(self, tokens, point, corruption_mask):
        return self.process.corrupt(tokens, point.coefficients, corruption_mask)

    def prior(self, tokens, point, prompt_mask):
        state = self.process.prior(len(tokens), tokens.shape[1], point.coefficients)
        return self.clamp_prompt(state, tokens, prompt_mask, point)

    def update(self, state, probabilities, source, destination, *, noise=None):
        del noise
        clean_tokens = torch.distributions.Categorical(probabilities).sample()
        return self.process.reverse(
            state, clean_tokens, destination.coefficients, source.coefficients, 1.0
        )

    def clamp_prompt(self, state, tokens, prompt_mask, point):
        del point
        clean = self.process.encode(tokens)
        return torch.where(prompt_mask.bool()[..., None], clean, state)
