"""Approximate rolling speculative decoding for embedding block diffusion."""

from __future__ import annotations

import math
import torch

from .base import Sampler, SamplerSpec, SamplingRequest, SamplingResult
from .registry import sampler_registry
from .utils import autoregressive_prompt_length


def _sample(logits, temperature, use_float64=False):
    if temperature == 0:
        return logits.argmax(-1)
    log_probs = (logits / temperature).log_softmax(-1)
    noise = -torch.empty_like(log_probs).exponential_().log()
    if use_float64:
        noise = noise.double()
    return (log_probs + noise).argmax(-1)


class BlockDiffusionSampler(Sampler):
    spec = SamplerSpec(
        name="block_diffusion",
        compatible_families=frozenset({"block_diffusion"}),
        required_capabilities=frozenset(
            {"block_diffusion_logits", "rolling_latents"}
        ),
    )

    def __init__(
        self,
        *,
        block_size,
        update_method,
        noise_correlation,
        temperature,
        use_float64,
        speculative=True,
        tokens_per_step=4,
        nfe_budget=None,
        warmup_steps=0,
        refinement_steps=0,
        refinement_projection="soft",
    ):
        if update_method not in {"euler", "renoise"}:
            raise ValueError("rolling_update must be 'euler' or 'renoise'")
        self.block_size = int(block_size)
        self.update_method = update_method
        self.noise_correlation = float(noise_correlation)
        self.temperature = float(temperature)
        self.use_float64 = bool(use_float64)
        self.speculative = bool(speculative)
        self.tokens_per_step = int(tokens_per_step)
        self.nfe_budget = None if nfe_budget is None else int(nfe_budget)
        self.warmup_steps = int(warmup_steps)
        self.refinement_steps = int(refinement_steps)
        self.refinement_projection = str(refinement_projection)
        if not self.speculative and self.tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be positive")
        if self.warmup_steps < 0:
            raise ValueError("rolling_warmup_steps must be non-negative")
        if self.refinement_steps < 0:
            raise ValueError("rolling_refinement_steps must be non-negative")
        if self.refinement_projection not in {"soft", "argmax"}:
            raise ValueError("rolling_x0_projection must be 'soft' or 'argmax'")
        if self.nfe_budget is not None and self.refinement_steps:
            raise ValueError(
                "nfe_budget with per-roll refinement is not yet supported; "
                "use fixed tokens_per_step"
            )

    @torch.no_grad()
    def sample(self, model, request: SamplingRequest) -> SamplingResult:
        batch = request.num_samples
        capacity = model.num_tokens
        tokens = torch.zeros(batch, capacity, dtype=torch.long, device=model.device)
        prompt_length = autoregressive_prompt_length(request)
        if prompt_length is None:
            tokens[:, 0] = model.tokenizer.bos_token_id
            boundary = 0
        else:
            tokens = torch.where(request.prompt_mask.bool(), request.prompt_tokens, tokens)
            boundary = max(prompt_length - 1, 0)

        window = min(self.block_size, capacity - boundary - 1)
        if window <= 0:
            return SamplingResult(tokens, {"nfe": 0, "accepted": 0, "rejected": 0})
        base_times = torch.linspace(0, 1, window, device=model.device)
        # Only the boundary anchor is known. Every future slot starts at the
        # flow prior t=1 and is moved onto the positional schedule in warmup.
        times = torch.ones(batch, window, device=model.device)
        times[:, 0] = 0
        anchor = model._normalized_embeddings(tokens[:, boundary : boundary + 1])
        latents = torch.randn(
            batch, window, anchor.shape[-1], device=model.device, dtype=anchor.dtype
        )
        latents[:, :1] = anchor

        nfe = accepted_total = rejected_total = 0
        generated_length = capacity - boundary - 1
        if not self.speculative and self.nfe_budget is not None:
            minimum_nfe = self.warmup_steps + math.ceil(
                generated_length / self.block_size
            )
            if not minimum_nfe <= self.nfe_budget <= generated_length:
                raise ValueError(
                    f"nfe_budget must be in [{minimum_nfe}, {generated_length}] "
                    f"for {generated_length} tokens and block size {self.block_size}"
                )

        # Initial rolling warmup: denoise unknown slots from t=1 toward their
        # position-dependent levels without moving the clean boundary.
        for warmup_index in range(self.warmup_steps):
            if window <= 1:
                break
            logits = model.active_window_logits(
                tokens[:, : boundary + 1], latents, times
            )
            x0_hat = model.logits_to_x0(
                logits[:, : window - 1], projection=self.refinement_projection
            )
            fraction = (warmup_index + 1) / self.warmup_steps
            target = 1 - fraction * (1 - base_times)
            next_times = target.expand(batch, -1).clone()
            next_times[:, 0] = 0
            latents[:, 1:] = model.update_rolling_latents(
                latents[:, 1:],
                x0_hat,
                times[:, 1:],
                next_times[:, 1:],
                method="euler",
                noise_correlation=0.0,
            )
            times = next_times
            nfe += 1
        while boundary < capacity - 1:
            window = min(self.block_size, capacity - boundary - 1)
            current_latents = latents[:, :window]
            current_times = times[:, :window]
            draft_logits = model.active_window_logits(
                tokens[:, : boundary + 1], current_latents, current_times
            )
            proposals = _sample(draft_logits, self.temperature, self.use_float64)
            if not self.speculative:
                nfe += 1
                remaining = capacity - boundary - 1
                if self.nfe_budget is None:
                    step = min(self.tokens_per_step, remaining, window)
                else:
                    remaining_nfe = self.nfe_budget - nfe + 1
                    step = math.ceil(remaining / remaining_nfe)
                    if step > window:
                        raise RuntimeError(
                            "remaining NFE budget is infeasible for the active window"
                        )
                committed = proposals
            else:
                verify_tokens = torch.cat(
                    (tokens[:, : boundary + 1], proposals), dim=1
                )
                verify_logits = model.clean_prefix_logits(verify_tokens)
                target_logits = verify_logits[:, boundary : boundary + window]
                nfe += 2

                if self.temperature == 0:
                    # Greedy decoding has no sampling distribution for the usual
                    # p/q correction; use point masses at each model's argmax.
                    q = torch.nn.functional.one_hot(
                        draft_logits.argmax(-1), draft_logits.shape[-1]
                    ).to(draft_logits.dtype)
                    p = torch.nn.functional.one_hot(
                        target_logits.argmax(-1), target_logits.shape[-1]
                    ).to(target_logits.dtype)
                else:
                    q = (draft_logits / self.temperature).softmax(-1)
                    p = (target_logits / self.temperature).softmax(-1)
                active = torch.ones(batch, dtype=torch.bool, device=model.device)
                advance = torch.zeros(batch, dtype=torch.long, device=model.device)
                committed = proposals.clone()
                for offset in range(window):
                    proposed = proposals[:, offset]
                    q_token = q[:, offset].gather(-1, proposed[:, None]).squeeze(-1)
                    p_token = p[:, offset].gather(-1, proposed[:, None]).squeeze(-1)
                    accept_prob = (p_token / q_token.clamp_min(1e-12)).clamp(max=1)
                    accepts = active & (torch.rand_like(accept_prob) < accept_prob)
                    rejects = active & ~accepts
                    if rejects.any():
                        residual = (p[:, offset] - q[:, offset]).clamp_min(0)
                        empty = residual.sum(-1) <= 1e-12
                        residual = torch.where(empty[:, None], p[:, offset], residual)
                        correction = _sample(
                            residual.clamp_min(1e-30).log(), 1.0, self.use_float64
                        )
                        committed[rejects, offset] = correction[rejects]
                        advance[rejects] = offset + 1
                        rejected_total += int(rejects.sum())
                    advance[accepts] = offset + 1
                    accepted_total += int(accepts.sum())
                    active = accepts
                # Batches generally advance different amounts. Commit the common
                # verified prefix; this keeps one dense rolling state per batch.
                step = max(int(advance.min().item()), 1)
            tokens[:, boundary + 1 : boundary + 1 + step] = committed[:, :step]
            boundary += step
            if boundary >= capacity - 1:
                break

            new_window = min(self.block_size, capacity - boundary - 1)
            target_times = torch.linspace(
                0, 1, new_window, device=model.device
            ).expand(batch, -1)
            # Fresh right-edge latents are samples from t=1, not samples from
            # their eventual positional time. Survivors alone can be moved to
            # lower noise because they have an x0 prediction.
            new_times = torch.ones(batch, new_window, device=model.device)
            new_times[:, 0] = 0
            new_anchor = model._normalized_embeddings(
                tokens[:, boundary : boundary + 1]
            )
            new_latents = torch.randn(
                batch,
                new_window,
                latents.shape[-1],
                device=latents.device,
                dtype=latents.dtype,
            )
            new_latents[:, :1] = new_anchor
            surviving = min(max(window - step - 1, 0), max(new_window - 1, 0))
            if surviving:
                old_slice = current_latents[:, step + 1 : step + 1 + surviving]
                old_times = current_times[:, step + 1 : step + 1 + surviving]
                if self.refinement_steps:
                    # Preserve the pre-shift state. Refinement passes below
                    # will transport it using predictions conditioned on the
                    # newly committed clean prefix.
                    new_latents[:, 1 : 1 + surviving] = old_slice
                    new_times[:, 1 : 1 + surviving] = old_times
                else:
                    # Legacy one-shot transport using the already available
                    # pre-commit predictions.
                    x0_hat = model.logits_to_x0(
                        draft_logits[:, step : step + surviving],
                        projection=self.refinement_projection,
                    )
                    new_latents[:, 1 : 1 + surviving] = model.update_rolling_latents(
                        old_slice,
                        x0_hat,
                        old_times,
                        target_times[:, 1 : 1 + surviving],
                        method=self.update_method,
                        noise_correlation=self.noise_correlation,
                    )
                    new_times[:, 1 : 1 + surviving] = target_times[
                        :, 1 : 1 + surviving
                    ]
            latents, times = new_latents, new_times

            # Settle the shifted window under the new clean prefix. Each pass
            # advances all unknown slots partway from their actual current
            # times (survivor time or fresh t=1) to the positional targets.
            for refinement_index in range(self.refinement_steps):
                if new_window <= 1:
                    break
                refinement_logits = model.active_window_logits(
                    tokens[:, : boundary + 1], latents, times
                )
                x0_hat = model.logits_to_x0(
                    refinement_logits[:, : new_window - 1],
                    projection=self.refinement_projection,
                )
                steps_left = self.refinement_steps - refinement_index
                next_times = times + (target_times - times) / steps_left
                next_times[:, 0] = 0
                latents[:, 1:] = model.update_rolling_latents(
                    latents[:, 1:],
                    x0_hat,
                    times[:, 1:],
                    next_times[:, 1:],
                    method="euler",
                    noise_correlation=0.0,
                )
                times = next_times
                nfe += 1

        return SamplingResult(
            tokens,
            {
                "nfe": nfe,
                "accepted": accepted_total,
                "rejected": rejected_total,
                "speculative": self.speculative,
                "tokens_per_step": None if self.speculative else self.tokens_per_step,
                "nfe_budget": None if self.speculative else self.nfe_budget,
                "warmup_steps": self.warmup_steps,
                "refinement_steps": self.refinement_steps,
                "refinement_projection": self.refinement_projection,
            },
        )


@sampler_registry.register("block_diffusion")
def _create_block_diffusion(config):
    return BlockDiffusionSampler(
        block_size=config.algo.block_size,
        update_method=config.algo.get("rolling_update", "euler"),
        noise_correlation=config.algo.get("renoise_correlation", 0.0),
        temperature=config.algo.get("temp", 1.0),
        use_float64=config.sampling.use_float64,
        warmup_steps=config.algo.get("rolling_warmup_steps", 0),
        refinement_steps=config.algo.get("rolling_refinement_steps", 0),
        refinement_projection=config.algo.get("rolling_x0_projection", "soft"),
    )


class VerifierFreeEulerSampler(BlockDiffusionSampler):
    spec = SamplerSpec(
        name="block_diffusion_euler",
        compatible_families=frozenset({"block_diffusion"}),
        required_capabilities=frozenset(
            {"block_diffusion_logits", "rolling_latents"}
        ),
    )


@sampler_registry.register("block_diffusion_euler")
def _create_block_diffusion_euler(config):
    return VerifierFreeEulerSampler(
        block_size=config.algo.block_size,
        update_method="euler",
        noise_correlation=0.0,
        temperature=config.algo.get("temp", 1.0),
        use_float64=config.sampling.use_float64,
        speculative=False,
        tokens_per_step=config.algo.get("euler_tokens_per_step", 4),
        nfe_budget=config.algo.get("euler_nfe_budget", None),
        warmup_steps=config.algo.get("rolling_warmup_steps", 0),
        refinement_steps=config.algo.get("rolling_refinement_steps", 0),
        refinement_projection=config.algo.get("rolling_x0_projection", "soft"),
    )
