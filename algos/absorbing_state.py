from .diffusion import Diffusion
from .trainer_base import _unsqueeze, sample_categorical


import torch
import torch.nn.functional as F


class AbsorbingState(Diffusion):
    sampling_family = "absorbing_diffusion"
    supported_samplers = frozenset({
        "ancestral", "ancestral_cache", "analytic", "entropy_bounded",
        "semi_ar",
    })
    sampling_capabilities = Diffusion.sampling_capabilities | {
        "ancestral_transition",
        "analytic_transition",
        "denoiser_transition",
        "zero_time_denoise",
        "denoising_log_probs",
    }
    def __init__(self, config, tokenizer):
        self.temp = config.algo.get("temp", 1.0)
        # NOTE: Ideally, we should do
        # vocab_size = len(tokenizer), so that we account
        # for the special tokens added in dataloader.py.
        # But we use tokenizer.vocab_size so as to to be
        # consistent with the prior checkpoints.
        vocab_size = tokenizer.vocab_size
        if hasattr(config.data, 'vocab_file'):
            vocab_size = len(tokenizer.added_tokens_decoder)
            self.token_offset = True
        else:
            self.token_offset = False
            if config.algo.get("mask_after_added_tokens", False):
                # Hugging Face's vocab_size excludes added special tokens.
                # Allocate the absorbing state after those IDs so it cannot
                # collide with an added PAD token (as in SmolLM/TinyGSM).
                vocab_size = max(vocab_size, len(tokenizer))
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id

        self.subs_masking = config.algo.subs_masking
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()

    def _validate_configuration(self):
        super()._validate_configuration()
        if self.parameterization in {"score", "mean"}:
            assert self.time_conditioning
        assert not (self.parameterization == "mean" and self.T == 0)
        if self.T > 0:
            assert self.parameterization in {"mean", "subs"}
        if self.subs_masking:
            assert self.parameterization == "mean"

    def q_xt(self, x, alpha_t, valid_tokens=None):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          alpha_t: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        xt = torch.where(move_indices, self.mask_index, x)
        if valid_tokens is not None:
            xt = torch.where(valid_tokens.bool(), xt, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        # pdb.set_trace()
        return self.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.device
        )

    def _ancestral_update(self, x, t, dt, p_x0=None, noise_removal_step=False, **kwargs):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2
        p_x0 = (self.forward(x, self._sigma_from_alphat(alpha_t)).double())
        p_x0 = p_x0.exp()
        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        _x = sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, dt):
        sigma_t = self._sigma_from_alphat(self.noise(t)[1])
        sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
        dsigma = sigma_t - sigma_s
        score = self._get_score(x, sigma_t)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        score = self._get_score(x, sigma)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[
            ..., None
        ]
        return edge
