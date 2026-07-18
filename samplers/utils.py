from __future__ import annotations

import torch
import torch.nn.functional as F


def inject_token_prompt(samples, request):
    if request.prompt_tokens is None or request.prompt_mask is None:
        return samples
    return torch.where(request.prompt_mask.bool(), request.prompt_tokens, samples)


def inject_onehot_prompt(samples, request):
    clean = torch.zeros(samples.shape[:2], device=samples.device, dtype=torch.bool)
    if request.prompt_tokens is None or request.prompt_mask is None:
        return samples, clean
    mask = request.prompt_mask.bool()
    prompt = F.one_hot(request.prompt_tokens, samples.shape[-1]).to(samples)
    return torch.where(mask.unsqueeze(-1), prompt, samples), mask


def diffusion_time_grid(model, request):
    steps = request.num_steps or model.config.sampling.steps
    times = torch.linspace(1, request.eps, steps + 1, device=model.device)
    return steps, times, (1 - request.eps) / steps


def continuous_time_grid(model, request):
    steps = request.num_steps or model.config.sampling.steps
    times = torch.linspace(0.999, request.eps, steps + 1, device=model.device)
    return steps, times


def apply_token_offset(model, samples):
    if model.token_offset:
        samples += model.tokenizer.eos_token_id
    return samples


def preserve_prompt(tokens, request):
    if request.prompt_tokens is None or request.prompt_mask is None:
        return tokens
    return torch.where(request.prompt_mask.bool(), request.prompt_tokens, tokens)


def run_continuous_diffusion(model, request):
    steps, times = continuous_time_grid(model, request)
    x, clean = inject_onehot_prompt(
        model.prior_sample(request.num_samples, model.num_tokens), request
    )
    sigmas = model.sampling_continuous_schedule(times)
    denoised = None
    for i in range(steps):
        x, denoised = model.sample_continuous_transition(
            x=x,
            time_t=times[i],
            time_s=times[i + 1],
            sigma_s=sigmas[i],
            sigma_t=sigmas[i + 1],
            clean_mask=clean,
        )
    return denoised.argmax(-1), steps
