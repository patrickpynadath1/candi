"""Likelihood bound over learned embeddings; every term is tokenwise FP64."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RePlaidTerms:
    prior: torch.Tensor
    reconstruction: torch.Tensor
    diffusion: torch.Tensor
    total: torch.Tensor


class RePlaidObjective:
    def __call__(
        self,
        log_probs,
        tokens,
        clean,
        predicted,
        gamma,
        gamma_prime,
        terminal_gamma,
        prior_clean,
        reconstruction_count,
    ):
        reconstruction = -log_probs.gather(-1, tokens[..., None]).squeeze(-1).double()
        snr_prime = -gamma_prime * torch.exp(-gamma)
        diffusion = (
            -0.5 * snr_prime[:, None] * (predicted - clean).double().square().sum(-1)
        )
        alpha_sq = torch.sigmoid(-terminal_gamma[0])
        sigma_sq = 1 - alpha_sq
        prior = 0.5 * (
            alpha_sq * prior_clean.double().square().sum(-1)
            + clean.shape[-1] * (sigma_sq - 1 - sigma_sq.log())
        )
        batch = len(tokens)
        recon = reconstruction_count
        if not 0 <= recon < batch:
            raise ValueError("allocation must include at least one diffusion example")
        weighted_recon = (
            reconstruction[:recon] * batch / recon if recon else reconstruction[:0]
        )
        weighted_diffusion = diffusion[recon:] * batch / (batch - recon)
        total = prior + torch.cat((weighted_recon, weighted_diffusion))
        return RePlaidTerms(prior, reconstruction, diffusion, total)
