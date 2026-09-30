"""Conditional best-of-K cross-entropy training from independent pure noise."""

import math

import torch
import torch.nn.functional as F

from .candi import CANDI
from .registry import algo_registry


def closest_candidate_loss(candidate_losses, valid_tokens):
    """Select one complete candidate per example, never per-token winners."""
    weights = valid_tokens.to(candidate_losses)
    scores = (candidate_losses.detach() * weights[None]).sum(-1)
    scores = scores / weights.sum(-1).clamp_min(1)[None]
    winners = scores.argmin(dim=0)
    batch = torch.arange(candidate_losses.shape[1], device=candidate_losses.device)
    return candidate_losses[winners, batch], winners


@algo_registry.register("candi_imle")
class CANDIIMLE(CANDI):
    default_sampler = "candi_imle"
    supported_samplers = frozenset({"candi_imle"})
    sampling_family = "candi_imle"
    sampling_capabilities = frozenset({"continuous_prior", "imle_prediction"})

    def _validate_configuration(self):
        if self.config.algo.backbone != "cont_dit":
            raise ValueError("candi_imle requires the standard cont_dit backbone")
        count = self.config.algo.imle_candidates
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("algo.imle_candidates must be a positive integer")
        sigma = float(self.config.algo.imle_noise_std)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("algo.imle_noise_std must be finite and positive")
        if self.config.noise.get("adaptive", False):
            raise ValueError("candi_imle uses fixed pure noise, not an adaptive clock")
        if self.config.algo.factorized_adaptive_schedule or self.config.algo.joint_loss_probe:
            raise ValueError("candi_imle does not use diffusion schedule probes")

    def prior_sample(self, *batch_dims):
        return torch.randn(
            *batch_dims, self.vocab_size - 1, device=self.device,
            dtype=self.backbone.vocab_embed.embedding.dtype,
        ) * float(self.config.algo.imle_noise_std)

    def sample_imle_distribution(self, state, reveal_mask):
        """One ordinary CANDI forward pass at the fixed pure-noise condition."""
        batch = state.shape[0]
        return self.forward(
            xt=state,
            reveal_mask=reveal_mask,
            discrete_noise=torch.ones(batch, device=state.device),
            continuous_noise=torch.full(
                (batch,), float(self.config.algo.imle_noise_std), device=state.device
            ),
        )

    def nll(self, x0, output_tokens, current_accumulation_step=None,
            train_mode=False, valid_tokens=None):
        del output_tokens, current_accumulation_step
        valid = (
            torch.ones_like(x0, dtype=torch.bool)
            if valid_tokens is None else valid_tokens.bool().clone()
        )
        if self.ignore_bos:
            valid[:, 0] = False
        reveal = ~valid
        # Target values are used only for CE. Every generated input position
        # is pure Gaussian noise, including for candidate selection.
        prompt = F.one_hot(x0.masked_fill(valid, 0), self.vocab_size - 1).to(
            self.backbone.vocab_embed.embedding
        )
        count = self.config.algo.imle_candidates if train_mode else 1
        losses = []
        for _ in range(count):
            noise = self.prior_sample(*x0.shape)
            state = torch.where(reveal[..., None], prompt, noise)
            log_probs = self.sample_imle_distribution(state, reveal)
            losses.append(self.raw_ce_per_token(log_probs, x0))
        # Retain the original graphs: dropout realizations used for selection
        # are exactly those differentiated by the winner's loss.
        selected, _ = closest_candidate_loss(torch.stack(losses), valid)
        return selected
