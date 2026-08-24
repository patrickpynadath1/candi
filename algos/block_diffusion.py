"""Continuous embedding block diffusion with an auxiliary clean AR stream."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ar import AR
from .registry import algo_registry


def block_diffusion_mask(length: int, block_size: int, device=None) -> torch.Tensor:
    """Mask for [noisy sequence, clean sequence], with True meaning visible."""
    if length <= 0 or block_size <= 0:
        raise ValueError("length and block_size must be positive")
    positions = torch.arange(length, device=device)
    query = positions[:, None]
    key = positions[None, :]
    causal = key <= query
    same_block = torch.div(key, block_size, rounding_mode="floor") == torch.div(
        query, block_size, rounding_mode="floor"
    )
    previous_block = torch.div(key, block_size, rounding_mode="floor") < torch.div(
        query, block_size, rounding_mode="floor"
    )

    mask = torch.zeros((2 * length, 2 * length), dtype=torch.bool, device=device)
    mask[:length, :length] = causal & same_block
    mask[:length, length:] = previous_block
    mask[length:, length:] = causal
    return mask


def sample_monotone_times(
    batch_size: int,
    length: int,
    block_size: int,
    *,
    max_time: float | torch.Tensor = 1.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Sample sorted-uniform position times with a sampled maximum per block."""
    if length <= 0 or block_size <= 0:
        raise ValueError("length and block_size must be positive")
    blocks = (length + block_size - 1) // block_size
    cap = torch.as_tensor(max_time, device=device, dtype=dtype)
    if cap.ndim == 0:
        cap = cap.expand(batch_size)
    if cap.shape != (batch_size,):
        raise ValueError("max_time must be scalar or [batch]")
    block_max = torch.rand(batch_size, blocks, device=device, dtype=dtype) * cap[:, None]
    values = []
    for block in range(blocks):
        width = min(block_size, length - block * block_size)
        if width == 1:
            values.append(block_max[:, block : block + 1])
            continue
        interior = torch.rand(
            batch_size, width - 1, device=device, dtype=dtype
        ).sort(dim=-1).values
        values.append(
            torch.cat((interior, torch.ones_like(interior[:, :1])), dim=-1)
            * block_max[:, block : block + 1]
        )
    return torch.cat(values, dim=-1)


def flow_interpolate(clean: torch.Tensor, times: torch.Tensor, noise=None):
    """Linear flow path z_t=(1-t)x_0+t*epsilon."""
    if noise is None:
        noise = torch.randn_like(clean)
    t = times.unsqueeze(-1).to(clean.dtype)
    return (1 - t) * clean + t * noise, noise


def rolling_latent_update(
    latent: torch.Tensor,
    x0_hat: torch.Tensor,
    old_time: torch.Tensor,
    new_time: torch.Tensor,
    *,
    method: str,
    noise_correlation: float = 0.0,
) -> torch.Tensor:
    """Move rolling latents to new position-dependent noise levels."""
    old_t = old_time.unsqueeze(-1).to(latent.dtype)
    new_t = new_time.unsqueeze(-1).to(latent.dtype)
    if method == "euler":
        eps_hat = (latent - (1 - old_t) * x0_hat) / old_t.clamp_min(1e-5)
        updated = (1 - new_t) * x0_hat + new_t * eps_hat
        return torch.where((old_t > 1e-5), updated, x0_hat)
    if method == "renoise":
        if not 0 <= noise_correlation <= 1:
            raise ValueError("noise_correlation must be in [0, 1]")
        eps_old = (latent - (1 - old_t) * x0_hat) / old_t.clamp_min(1e-5)
        eps_old = torch.where(old_t > 1e-5, eps_old, torch.randn_like(latent))
        fresh = torch.randn_like(latent)
        rho = noise_correlation
        eps = rho * eps_old + (1 - rho**2) ** 0.5 * fresh
        return (1 - new_t) * x0_hat + new_t * eps
    raise ValueError("method must be 'euler' or 'renoise'")


@algo_registry.register("block_diffusion")
class BlockDiffusion(AR):
    default_sampler = "block_diffusion"
    supported_samplers = frozenset({"block_diffusion", "block_diffusion_euler"})
    sampling_family = "block_diffusion"
    sampling_capabilities = frozenset({"block_diffusion_logits", "rolling_latents"})

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.block_size = int(config.algo.block_size)
        self.noise_curriculum_steps = int(config.algo.get("noise_curriculum_steps", 0))
        self.max_noise_level = float(config.algo.get("max_noise_level", 1.0))
        self.clean_loss_weight = float(config.algo.get("clean_loss_weight", 0.5))
        self.gradient_cosine_every = int(
            config.algo.get("gradient_cosine_every", 0)
        )
        self.embedding_scale = float(config.algo.get("embedding_scale", 1.0))
        self.save_hyperparameters()

    def _validate_configuration(self):
        super()._validate_configuration()
        if not self.config.algo.causal_attention:
            raise ValueError("block diffusion requires causal_attention=true")
        if int(self.config.algo.block_size) <= 0:
            raise ValueError("block_size must be positive")
        clean_weight = float(self.config.algo.get("clean_loss_weight", 0.5))
        if not 0 <= clean_weight <= 1:
            raise ValueError("clean_loss_weight must be in [0, 1]")

    def _normalized_embeddings(self, tokens):
        embeddings = self.backbone.get_embedding(tokens)
        return F.normalize(embeddings, dim=-1) * self.embedding_scale

    def _curriculum_cap(self, train_mode):
        if not train_mode or self.noise_curriculum_steps <= 0:
            return self.max_noise_level
        progress = min(float(self.global_step) / self.noise_curriculum_steps, 1.0)
        return self.max_noise_level * progress

    def doubled_logits(self, clean_tokens, times, noise=None):
        clean = self._normalized_embeddings(clean_tokens)
        noisy, noise = flow_interpolate(clean, times, noise)
        embeddings = torch.cat((noisy, clean), dim=1)
        length = clean_tokens.shape[1]
        positions = torch.arange(length, device=clean_tokens.device).repeat(2)
        positions = positions.expand(clean_tokens.shape[0], -1)
        mask = block_diffusion_mask(length, self.block_size, clean_tokens.device)
        noise_times = torch.cat((times, torch.zeros_like(times)), dim=1)
        logits = self.backbone.forward_masked_embeddings(
            embeddings, mask, positions, noise_times=noise_times
        )
        return logits[:, :length], logits[:, length:], noisy, noise

    def nll(
        self,
        input_tokens,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        del current_accumulation_step
        times = sample_monotone_times(
            input_tokens.shape[0],
            input_tokens.shape[1],
            self.block_size,
            max_time=self._curriculum_cap(train_mode),
            device=input_tokens.device,
            dtype=self.backbone.get_embedding(input_tokens).dtype,
        )
        noisy_logits, clean_logits, _, _ = self.doubled_logits(input_tokens, times)
        if self.mask_index is not None:
            noisy_logits[..., self.mask_index] = self.neg_infinity
            clean_logits[..., self.mask_index] = self.neg_infinity
        noisy_nll = F.cross_entropy(
            noisy_logits.transpose(1, 2), output_tokens, reduction="none"
        )
        clean_nll = F.cross_entropy(
            clean_logits.transpose(1, 2), output_tokens, reduction="none"
        )
        if valid_tokens is None:
            valid_tokens = torch.ones_like(clean_nll)
        denominator = valid_tokens.sum().clamp_min(1)
        self._last_clean_loss = (clean_nll * valid_tokens).sum() / denominator
        self._last_noisy_loss = (noisy_nll * valid_tokens).sum() / denominator
        return (
            self.clean_loss_weight * clean_nll
            + (1 - self.clean_loss_weight) * noisy_nll
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log(
            "diagnostics/clean_nll",
            self._last_clean_loss.detach(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "diagnostics/noisy_nll",
            self._last_noisy_loss.detach(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        if self.gradient_cosine_every > 0 and (
            int(self.global_step) % self.gradient_cosine_every == 0
        ):
            parameters = [
                parameter
                for parameter in self.backbone.parameters()
                if parameter.requires_grad
            ]
            clean_gradients = torch.autograd.grad(
                self._last_clean_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            noisy_gradients = torch.autograd.grad(
                self._last_noisy_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            dot = torch.zeros((), device=self.device, dtype=torch.float32)
            clean_norm_sq = torch.zeros_like(dot)
            noisy_norm_sq = torch.zeros_like(dot)
            for clean_gradient, noisy_gradient in zip(
                clean_gradients, noisy_gradients
            ):
                if clean_gradient is None or noisy_gradient is None:
                    continue
                clean_gradient = clean_gradient.detach().float()
                noisy_gradient = noisy_gradient.detach().float()
                dot += (clean_gradient * noisy_gradient).sum()
                clean_norm_sq += clean_gradient.square().sum()
                noisy_norm_sq += noisy_gradient.square().sum()
            clean_norm = clean_norm_sq.sqrt()
            noisy_norm = noisy_norm_sq.sqrt()
            cosine = dot / (clean_norm * noisy_norm).clamp_min(1e-12)
            self.log(
                "diagnostics/clean_noisy_grad_cosine",
                cosine,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                "diagnostics/clean_grad_norm",
                clean_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                "diagnostics/noisy_grad_norm",
                noisy_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        return loss

    def block_diffusion_logits(self, clean_tokens, times, noise=None):
        return self.doubled_logits(clean_tokens, times, noise)

    def clean_prefix_logits(self, tokens):
        embeddings = self._normalized_embeddings(tokens)
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).expand(
            tokens.shape[0], -1
        )
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        times = torch.zeros_like(tokens, dtype=embeddings.dtype)
        logits = self.backbone.forward_masked_embeddings(
            embeddings, mask, positions, noise_times=times
        )
        if self.mask_index is not None:
            logits[..., self.mask_index] = self.neg_infinity
        return logits

    def active_window_logits(self, clean_prefix, noisy_window, noisy_times):
        """Predict the next token from each source in a rolling noisy window."""
        batch, window, _ = noisy_window.shape
        prefix = self._normalized_embeddings(clean_prefix)
        prefix_length = clean_prefix.shape[1]
        combined = torch.cat((noisy_window, prefix), dim=1)
        if noisy_times.shape != noisy_window.shape[:2]:
            raise ValueError("noisy_times must match the rolling window")
        combined_times = torch.cat(
            (noisy_times, torch.zeros_like(clean_prefix, dtype=noisy_times.dtype)),
            dim=1,
        )
        noisy_positions = torch.arange(
            prefix_length - 1,
            prefix_length - 1 + window,
            device=clean_prefix.device,
        )
        clean_positions = torch.arange(prefix_length, device=clean_prefix.device)
        positions = torch.cat((noisy_positions, clean_positions)).expand(batch, -1)

        total = window + prefix_length
        mask = torch.zeros(total, total, dtype=torch.bool, device=clean_prefix.device)
        mask[:window, :window] = torch.ones(
            window, window, dtype=torch.bool, device=clean_prefix.device
        ).tril()
        # The boundary token exists in the noisy stream; only earlier clean
        # tokens are visible, preventing duplicate access to that position.
        if prefix_length > 1:
            mask[:window, window : window + prefix_length - 1] = True
        mask[window:, window:] = torch.ones(
            prefix_length,
            prefix_length,
            dtype=torch.bool,
            device=clean_prefix.device,
        ).tril()
        logits = self.backbone.forward_masked_embeddings(
            combined, mask, positions, noise_times=combined_times
        )
        logits = logits[:, :window]
        if self.mask_index is not None:
            logits[..., self.mask_index] = self.neg_infinity
        return logits

    def logits_to_x0(self, logits, projection="soft"):
        # Attention commonly emits BF16 logits at inference while learned
        # embeddings remain FP32. Accumulate the expected clean embedding in
        # FP32 both for dtype compatibility and for stable rolling updates.
        embedding_table = F.normalize(
            self.backbone.vocab_embed.embedding.float(), dim=-1
        )
        if projection == "argmax":
            return embedding_table[logits.argmax(-1)] * self.embedding_scale
        if projection == "soft":
            probs = logits.float().softmax(-1)
            return (
                torch.einsum("...v,vd->...d", probs, embedding_table)
                * self.embedding_scale
            )
        raise ValueError("projection must be 'soft' or 'argmax'")

    def update_rolling_latents(self, *args, **kwargs):
        return rolling_latent_update(*args, **kwargs)
