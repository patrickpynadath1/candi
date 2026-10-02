"""Posterior-derived tangent flow and manifold integration."""

import torch
from core.processes.spherical import log_map
from core.modifiers.self_conditioning.session import prediction_session


def posterior_velocity(state, embeddings, probabilities, mode="exact", eps=1e-6):
    if mode == "sample":
        indices = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), 1
        ).reshape(probabilities.shape[:-1])
        return log_map(state, embeddings[indices], eps)
    if mode != "exact":
        raise ValueError("velocity must be exact or sample")
    cosine = torch.einsum("bld,vd->blv", state, embeddings).clamp(-1, 1)
    angle = torch.acos(cosine.clamp(-1 + eps, 1 - eps))
    scaled = probabilities * angle / angle.sin().clamp_min(eps)
    return torch.einsum("blv,vd->bld", scaled, embeddings) - state * (
        scaled * cosine
    ).sum(-1, keepdim=True)


class SphericalFlowSampler:
    def __init__(self, steps=180, epsilon=1e-5, velocity="exact", use_float64=True):
        if steps < 1 or not 0 < epsilon < 1 or velocity not in {"exact", "sample"}:
            raise ValueError("invalid spherical integration settings")
        self.steps, self.epsilon, self.velocity = steps, epsilon, velocity
        self.use_float64 = use_float64

    def sample(self, algo, prompt_tokens, prompt_mask):
        from core.algorithms.spherical_flow_matching import SphericalFlowMatching

        if not isinstance(algo, SphericalFlowMatching):
            raise TypeError("spherical sampler requires spherical flow matching")
        if prompt_tokens.ndim != 2 or prompt_mask.shape != prompt_tokens.shape:
            raise ValueError(
                "prompt tokens and mask must have matching [batch, length] shapes"
            )
        algo = prediction_session(algo)
        clean = algo.representation.encode(prompt_tokens)
        mask = prompt_mask.bool().unsqueeze(-1)
        state = torch.where(mask, clean, algo.process.prior(clean))
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=state.device)
        dtype = (
            torch.float64
            if algo.process.precision == "float64" or self.use_float64
            else torch.float32
        )
        embeddings = algo.representation.codebook().to(dtype)
        for index in range(self.steps):
            keep, _ = algo.schedule(times[index].expand(len(state)))
            prediction = algo.sampling_prediction(state, keep)
            if self.use_float64:
                prediction = prediction.double()
            if index == self.steps - 1:
                break
            probabilities = algo.predictor.probabilities(
                prediction.to(dtype).log_softmax(-1)
            )
            velocity = posterior_velocity(
                state.to(dtype),
                embeddings,
                probabilities,
                self.velocity,
                algo.process.eps,
            )
            next_keep, _ = algo.schedule(times[index + 1].expand(len(state)))
            delta = ((next_keep - keep) / (1 - keep).clamp_min(algo.process.eps))[
                :, None, None
            ]
            state = algo.process.advance(state.to(dtype), delta * velocity).to(
                clean.dtype
            )
            state = torch.where(mask, clean, state)
        probabilities = algo.predictor.probabilities(prediction, apply_commitment=True)
        tokens = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), 1
        ).reshape(prompt_tokens.shape)
        return torch.where(prompt_mask.bool(), prompt_tokens, tokens), {
            "nfe": self.steps,
            "nfe_per_sample": [self.steps] * len(tokens),
        }
