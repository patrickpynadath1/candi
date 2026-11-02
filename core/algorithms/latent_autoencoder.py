"""Clean token reconstruction plus continuous latent denoising.

Decoded denoising CE is detached feedback for a time sampler, never a denoiser
training loss. Gradient routing to the encoder is explicit and configurable.
"""

import torch
from torch import nn


class LatentAutoencoderDiffusion(nn.Module):
    validation_metric = "val/latent_objective"

    def __init__(
        self,
        model,
        process,
        schedule,
        time_sampler,
        denoising_loss,
        clean_loss,
        clean_loss_weight=1.0,
        denoising_loss_weight=1.0,
        detach_encoder_for_denoising=True,
    ):
        super().__init__()
        if clean_loss_weight <= 0 or denoising_loss_weight <= 0:
            raise ValueError("both loss weights must be positive")
        self.model, self.process, self.schedule = model, process, schedule
        self.time_sampler = time_sampler
        self.denoising_loss, self.clean_loss = denoising_loss, clean_loss
        self.clean_loss_weight, self.denoising_loss_weight = (
            clean_loss_weight,
            denoising_loss_weight,
        )
        self.detach_encoder_for_denoising = detach_encoder_for_denoising

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        **time_context,
    ):
        mask = loss_mask.bool() if corruption_mask is None else corruption_mask.bool()
        # Within-block encoding mixes tokens, so partially revealed blocks are unsafe.
        changes = mask[:, 1:] != mask[:, :-1]
        interior = (
            torch.arange(1, tokens.shape[1], device=tokens.device)
            % self.model.block_size
            != 0
        )
        if (changes & interior).any():
            raise ValueError("contextual latent corruption requires whole blocks")
        if not loss_mask.any():
            raise ValueError("loss mask must include targets")
        clean = self.model.encode(tokens)
        clean_logits = self.model.decode(clean, clean)
        reconstruction = self.clean_loss(clean_logits.float().log_softmax(-1), tokens)
        target = clean.detach() if self.detach_encoder_for_denoising else clean
        times = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        path = self.schedule(times)
        noisy = self.process.corrupt_state(target, path, mask)
        estimate = self.model.denoise(noisy, times, target)
        denoising = self.denoising_loss(estimate, target)
        weights = loss_mask.to(denoising)
        count = weights.sum()
        mse = (denoising * weights).sum() / count
        ce = (reconstruction * weights).sum() / count
        # The token decoder trains on clean representations, not denoiser outputs.
        with torch.no_grad():
            decoded = self.model.decode(estimate.detach(), clean.detach())
            token_ce = self.clean_loss(decoded.float().log_softmax(-1), tokens)
            observe = getattr(self.time_sampler, "observe", None)
            if training and observe is not None:
                observe(times, token_ce, loss_mask, step)
        loss = self.denoising_loss_weight * mse + self.clean_loss_weight * ce
        return dict(
            loss=loss,
            loss_sum=loss.detach() * count,
            num_tokens=count.detach(),
            latent_mse=mse,
            clean_decoding_loss=ce,
            decoded_denoising_ce=(token_ce * weights).sum() / count,
        )

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)
