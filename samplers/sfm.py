"""Manifold ODE sampler for Spherical Flow Matching."""

import torch

from algos.spherical_utils import exp_map, log_map
from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


def sample_categorical(probs):
    return (probs / torch.empty_like(probs).exponential_().clamp_min(1e-20)).argmax(-1)


def sfm_compute_velocity(x, embeddings, log_p, mode="exact", eps=1e-6):
    p = log_p.exp()
    if mode == "sample":
        return log_map(x, embeddings[sample_categorical(p)], eps)
    cosine = torch.einsum("bld,vd->blv", x, embeddings).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cosine)
    scaled = p * omega / omega.sin().clamp_min(eps)
    return torch.einsum("blv,vd->bld", scaled, embeddings) - x * (scaled * cosine).sum(
        -1, keepdim=True
    )


def sfm_step_size(alpha_t, alpha_s, invert=True, eps=1e-6):
    return (
        (alpha_t - alpha_s) / alpha_t.clamp_min(eps)
        if invert
        else (alpha_s - alpha_t) / (1 - alpha_t).clamp_min(eps)
    )


class SFMSampler(Sampler):
    spec = SamplerSpec(
        "sfm",
        frozenset({"sfm"}),
        frozenset(
            {
                "spherical_prior",
                "spherical_schedule",
                "spherical_embeddings",
                "spherical_logits",
            }
        ),
    )

    def __init__(self, config):
        s = config.sampling
        a = config.algo
        self.velocity = getattr(s, "velocity", "exact")
        self.top_k_velocity = getattr(s, "top_k_velocity", -1)
        self.use_float64 = getattr(s, "use_float64", False)
        self.noise_removal = getattr(s, "noise_removal", "ancestral")
        self.invert = getattr(a, "invert_time_convention", True)
        self.eps = getattr(a, "eps", 1e-6)

    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        x = model.prior_sample(request.num_samples, model.num_tokens)
        times = (
            torch.linspace(request.eps, 1, steps + 1, device=model.device)
            if self.invert
            else torch.linspace(1, request.eps, steps + 1, device=model.device)
        )
        E = model.sampling_embeddings()
        prompt_emb = (
            model.backbone.get_sphere_embeddings(request.prompt_tokens)
            if request.prompt_tokens is not None
            else None
        )
        mask = request.prompt_mask.bool() if request.prompt_mask is not None else None
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), prompt_emb, x)
        for i in range(steps):
            t = times[i].expand(request.num_samples)
            logp = model.sampling_logits(x, t)
            if i == steps - 1:
                break
            px = logp
            if self.top_k_velocity > 0:
                vals, idx = torch.topk(px, self.top_k_velocity, -1)
                local_E = E[idx]
                p = vals.log_softmax(-1)
                chosen = sample_categorical(p)
                target = (
                    torch.gather(
                        local_E,
                        -2,
                        chosen[..., None, None].expand(*chosen.shape, 1, E.shape[-1]),
                    ).squeeze(-2)
                    if self.velocity == "sample"
                    else None
                )
                if self.velocity == "sample":
                    vel = log_map(x, target, self.eps)
                else:
                    cos = (
                        (x.unsqueeze(-2) * local_E)
                        .sum(-1)
                        .clamp(-1 + self.eps, 1 - self.eps)
                    )
                    om = torch.acos(cos)
                    sc = p.exp() * om / om.sin().clamp_min(self.eps)
                    vel = (sc.unsqueeze(-1) * local_E).sum(-2) - x * (sc * cos).sum(
                        -1, keepdim=True
                    )
            else:
                vel = sfm_compute_velocity(
                    x, E, px.log_softmax(-1), self.velocity, self.eps
                )
            at = model.sampling_alpha(times[i])
            ans = model.sampling_alpha(times[i + 1])
            x = exp_map(
                x, sfm_step_size(at, ans, self.invert, self.eps) * vel, self.eps
            )
            if mask is not None:
                x = torch.where(mask.unsqueeze(-1), prompt_emb, x)
        tokens = (
            logp.argmax(-1)
            if self.noise_removal == "greedy"
            else sample_categorical(logp.softmax(-1))
        )
        return SamplingResult(preserve_prompt(tokens, request), {"nfe": steps})


@sampler_registry.register("sfm")
def _create(config):
    return SFMSampler(config)
