"""GIDD mask/uniform corruption and its marginal plug-in reverse transition."""

import torch
from torch.nn import functional as F

from core.samplers.utils import sample_categorical


class GIDDProcess:
    # Hybrid prediction is a clean-token distribution. The update below follows
    # GIDD's marginal plug-in rule, not PosteriorUniformProcess's bridge mixture.
    prediction_kind = "posterior"

    def __init__(self, vocab_size):
        if vocab_size < 2:
            raise ValueError("GIDD requires at least two clean tokens")
        self.vocab_size = vocab_size
        self.mask_index = vocab_size

    def mixing(self, path):
        uniform = path.uniform[:, None].expand(-1, self.vocab_size) / self.vocab_size
        return torch.cat((uniform, path.mask[:, None]), dim=-1)

    def mixing_derivative(self, path):
        uniform = path.duniform[:, None].expand(-1, self.vocab_size) / self.vocab_size
        return torch.cat((uniform, path.dmask[:, None]), dim=-1)

    def marginal(self, clean_probabilities, path):
        if clean_probabilities.shape[-1] != self.vocab_size:
            raise ValueError(
                "GIDD expects a clean-token distribution excluding its mask"
            )
        return (
            path.alpha[:, None, None] * F.pad(clean_probabilities.double(), (0, 1))
            + self.mixing(path)[:, None, :]
        )

    def forward_distribution(self, clean, coefficients):
        return torch.distributions.Categorical(probs=self.marginal(clean, coefficients))

    def corrupt_state(self, clean, coefficients, corruption_mask):
        probs = F.one_hot(clean, self.vocab_size)
        noisy = sample_categorical(self.marginal(probs, coefficients))
        return torch.where(corruption_mask.bool(), noisy, clean)

    def prior_state(self, clean, coefficients):
        # The terminal marginal is all-mask, even when uniform noise is present
        # along the path. Use this prior only at the configured terminal endpoint.
        return torch.full_like(clean, self.mask_index)

    def reverse_probabilities(self, state, clean_probabilities, source, destination):
        if clean_probabilities.shape != (*state.shape, self.vocab_size):
            raise ValueError("prediction must have shape [batch, length, clean_vocab]")
        if (
            source.alpha.shape != state.shape[:1]
            or destination.alpha.shape != source.alpha.shape
        ):
            raise ValueError("path coefficients must have shape [batch]")
        if torch.any(destination.alpha < source.alpha):
            raise ValueError("reverse updates must move toward the clean endpoint")
        observed = F.one_hot(state, self.vocab_size + 1).double()
        same = source.alpha == destination.alpha
        ratio = torch.where(same, 1, source.alpha / destination.alpha.clamp_min(1e-300))
        mixing = self.mixing(source) - ratio[:, None] * self.mixing(destination)
        if torch.any(mixing < -1e-10):
            raise ValueError("schedule gives negative conditional mixing probabilities")
        mixing = mixing.clamp_min(0)
        likelihood = ratio[:, None, None] * observed + mixing[:, None, :].expand_as(
            observed
        ).gather(-1, state[..., None])
        previous = self.marginal(clean_probabilities, destination)
        current = self.marginal(clean_probabilities, source).gather(
            -1, state[..., None]
        )
        # Clamped prompt tokens can be outside the marginal support (a visible
        # token at the all-mask endpoint). Their conditional is undefined;
        # preserve them rather than producing NaNs before the sampler clamps.
        preserve = same[:, None, None] | (current <= 0)
        result = likelihood * previous / current.clamp_min(1e-300)
        return torch.where(preserve, observed, result)

    def reverse(self, state, clean_probabilities, source, destination):
        return sample_categorical(
            self.reverse_probabilities(state, clean_probabilities, source, destination)
        )
