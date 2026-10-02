"""Absorbing-state corruption and executable reverse transitions."""

import torch
from core.samplers.utils import sample_categorical


class AbsorbingProcess:
    def __init__(self, mask_index):
        self.mask_index = mask_index

    def prior(self, shape, device, prompt_tokens=None, prompt_mask=None):
        state = torch.full(shape, self.mask_index, dtype=torch.long, device=device)
        if prompt_tokens is not None:
            state = torch.where(prompt_mask.bool(), prompt_tokens, state)
        return state

    def corrupt(self, clean, keep_rate, corruption_mask):
        moves = torch.rand(clean.shape, device=clean.device) < 1 - keep_rate[:, None]
        return torch.where(moves & corruption_mask.bool(), self.mask_index, clean)

    def reverse(self, state, posterior, keep_t, keep_s):
        if torch.any((keep_t < 0) | (keep_s > 1) | (keep_s < keep_t)):
            raise ValueError("reverse transition requires 0 <= keep_t <= keep_s <= 1")
        if torch.any((keep_t >= 1) & state.eq(self.mask_index).any(-1)):
            raise ValueError("a clean source cannot contain absorbing masks")
        # Joint categorical matches the existing native MDLM reverse kernel
        # and its random-number consumption, rather than two separate draws.
        probabilities = posterior.double() * (keep_s - keep_t)[:, None, None]
        probabilities[..., self.mask_index] = (1 - keep_s)[:, None]
        proposed = sample_categorical(probabilities)
        return torch.where(state.eq(self.mask_index), proposed, state)

    def commit(self, state, posterior, selected, *, deterministic=False):
        # EB uses the same multinomial draw as the legacy implementation.
        if deterministic:
            proposed = posterior.argmax(-1)
        else:
            proposed = torch.multinomial(
                posterior.float().reshape(-1, posterior.shape[-1]), 1
            )
            proposed = proposed.reshape(state.shape)
        return torch.where(selected & state.eq(self.mask_index), proposed, state)

    def staggered_score(self, score, delta_sigma):
        score = score.clone()
        extra = (1 - delta_sigma.exp()[:, None]) * score.sum(-1)
        score *= delta_sigma.exp()[:, None, None]
        score[..., self.mask_index] += extra
        return score

    def transposed_transition(self, state, delta_sigma, vocab_size):
        decay = (-delta_sigma).exp()[:, None, None]
        edge = decay * torch.nn.functional.one_hot(state, vocab_size)
        edge += torch.where(state.eq(self.mask_index)[..., None], 1 - decay, 0)
        return edge

    def analytic_probabilities(self, state, score, delta_sigma, *, denoise=False):
        probabilities = self.staggered_score(
            score, delta_sigma
        ) * self.transposed_transition(state, delta_sigma, score.shape[-1])
        if denoise:
            probabilities[..., self.mask_index] = 0
        return probabilities

    def analytic_reverse(self, state, score, delta_sigma, *, denoise=False):
        return sample_categorical(
            self.analytic_probabilities(state, score, delta_sigma, denoise=denoise)
        )
