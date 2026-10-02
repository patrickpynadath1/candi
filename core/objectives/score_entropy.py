"""Per-token legacy diffusion objective, isolated from training orchestration."""

import torch


class AbsorbingScoreEntropy:
    def __init__(self, mask_index, vocab_size):
        self.mask_index, self.vocab_size = mask_index, vocab_size

    def __call__(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        """Score entropy at masked positions, including the reference constant."""
        assert not low_var
        masked_indices = xt == self.mask_index
        sigma = -alpha_t.log()
        dsigma = -dalpha_t / alpha_t
        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]
        words_that_were_masked = x0[masked_indices]
        neg_term = q_ratio * torch.gather(
            log_x_theta[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_x_theta[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(dim=-1) + score[
                :, self.mask_index + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)
        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return dsigma * entropy
