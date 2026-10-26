"""Continuous map components, with framework time 1=noise and 0=data.

Linear mean-denoiser transport and PSD: arxiv:2604.09784, Eqs. 20, 27;
FMLM direct-training variant: arxiv:2602.16813, Algorithm 7 / Appendix C.
Beckmann directional target: arxiv:2609.15903, Eq. 17. These components do
not implement paper-specific schedules, anchor curricula or quality heads.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from core.models.layers import TimestepEmbedder


def _time(time, state):
    time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
    if time.ndim == 0:
        time = time.expand(state.shape[0])
    if time.shape != (state.shape[0],):
        raise ValueError("map times must be scalar or batch-shaped")
    if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError("map times must be finite and within [0, 1]")
    return time


class LinearInterpolation:
    """One shared Gaussian draw couples all times; velocity points toward data."""

    def __init__(self, noise_scale=1.0):
        if not 0 < noise_scale < float("inf"):
            raise ValueError("noise_scale must be positive and finite")
        self.noise_scale = float(noise_scale)

    def sample(self, tokens, vocab_size, time, noise=None, mask=None):
        clean = F.one_hot(tokens, vocab_size).float()
        noise = torch.randn_like(clean) * self.noise_scale if noise is None else noise
        if noise.shape != clean.shape:
            raise ValueError("noise and one-hot state shapes must match")
        t = _time(time, clean)[:, None, None]
        state = (1 - t) * clean + t * noise
        velocity = clean - noise  # derivative wrt paper's forward progress 1-t
        if mask is not None:
            state = torch.where(mask[..., None].bool(), state, clean)
            velocity = torch.where(mask[..., None].bool(), velocity, 0)
        return state, velocity


def linear_transport(state, probabilities, source, destination):
    """Finite map, including exact identity at the clean diagonal."""
    source, destination = _time(source, state), _time(destination, state)
    if (destination > source).any():
        raise ValueError("transport requires destination <= source")
    ratio = torch.where(source > 0, destination / source.clamp_min(1e-30), 1)
    ratio = ratio[:, None, None]
    return ratio * state + (1 - ratio) * probabilities


class MeanDenoiserMap(nn.Module):
    """Two-time categorical map on the existing DiT backbone execution."""

    def __init__(self, backbone, vocab_size):
        super().__init__()
        self.backbone = getattr(backbone, "backbone", backbone)
        if not self.backbone.adaLN:
            raise ValueError("two-time map requires an adaLN backbone")
        hidden = self.backbone.blocks[0].attn_qkv.in_features
        cond_dim = self.backbone.sigma_map.mlp[-1].out_features
        if not hasattr(self.backbone, "vocab_embed"):
            raise ValueError("map adapters require a DiT backbone with a token table")
        if self.backbone.vocab_embed.embedding.shape != (vocab_size, hidden):
            raise ValueError(
                "backbone input table must match vocabulary and hidden size"
            )
        self.destination_time = TimestepEmbedder(cond_dim)
        self.vocab_size = int(vocab_size)

    def forward(self, state, source, destination):
        source, destination = _time(source, state), _time(destination, state)
        if (destination > source).any():
            raise ValueError("map requires destination <= source")
        table = self.backbone.vocab_embed.embedding
        hidden = state.to(table.dtype) @ table
        condition = self.backbone.time_adapter(source, self.backbone.sigma_map)
        condition = condition + self.backbone.time_adapter(
            destination, self.destination_time
        )
        logits = self.backbone.forward_hidden(hidden, condition)
        if logits.shape[-1] != self.vocab_size:
            raise ValueError("backbone output must match the one-hot vocabulary")
        return logits.float().log_softmax(-1)

    def transport(self, state, source, destination):
        return linear_transport(
            state, self(state, source, destination).exp(), source, destination
        )


class EndpointMap(nn.Module):
    """Autonomous categorical endpoint: no time input reaches the network."""

    def __init__(self, backbone, vocab_size):
        super().__init__()
        self.backbone = getattr(backbone, "backbone", backbone)
        hidden = self.backbone.blocks[0].attn_qkv.in_features
        if not hasattr(self.backbone, "vocab_embed"):
            raise ValueError("map adapters require a DiT backbone with a token table")
        if self.backbone.vocab_embed.embedding.shape != (vocab_size, hidden):
            raise ValueError(
                "backbone input table must match vocabulary and hidden size"
            )
        self.vocab_size = int(vocab_size)

    def forward(self, state):
        table = self.backbone.vocab_embed.embedding
        hidden = state.to(table.dtype) @ table
        condition = None
        if self.backbone.adaLN:
            # Constant conditioning retains the backbone architecture, not time.
            condition = self.backbone.time_adapter(
                torch.zeros(len(state), device=state.device), self.backbone.sigma_map
            )
        logits = self.backbone.forward_hidden(hidden, condition)
        if logits.shape[-1] != self.vocab_size:
            raise ValueError("backbone output must match the one-hot vocabulary")
        return logits.float().log_softmax(-1)


class SemigroupTarget:
    """Exact linear mean-denoiser PSD target, not the composed state itself."""

    @torch.no_grad()
    def __call__(
        self,
        predictor,
        state,
        source,
        middle,
        destination,
        prompt_state=None,
        prompt_mask=None,
    ):
        source, middle, destination = (
            _time(t, state) for t in (source, middle, destination)
        )
        if not ((source > middle) & (middle > destination)).all():
            raise ValueError("PSD requires source > middle > destination")
        first = predictor(state, source, middle).exp()
        intermediate = linear_transport(state, first, source, middle)
        if (prompt_state is None) != (prompt_mask is None):
            raise ValueError("prompt_state and prompt_mask must be supplied together")
        if prompt_mask is not None:
            intermediate = torch.where(
                prompt_mask[..., None].bool(), prompt_state, intermediate
            )
        second = predictor(intermediate, middle, destination).exp()
        weight = destination * (source - middle) / (middle * (source - destination))
        return weight[:, None, None] * first + (1 - weight[:, None, None]) * second


class DirectionalEndpointTarget:
    """Beckmann Eq. 17. Signed target for squared loss; never softmax it.

    velocity is dI/d(progress), i.e. clean-noise for a linear path. eval mode
    is required to give target/JVP a deterministic function (dropout is invalid).
    """

    def __call__(self, endpoint, state, velocity):
        if state.shape != velocity.shape:
            raise ValueError("state and directional velocity shapes must match")
        if isinstance(endpoint, nn.Module) and endpoint.training:
            raise ValueError("directional targets require an eval-mode endpoint")
        # The fused SDPA backward does not support the second derivative used
        # by autograd.functional.jvp. Limit the math backend to this call.
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            value, tangent = torch.autograd.functional.jvp(
                lambda x: endpoint(x).exp(), state, velocity, create_graph=False
            )
        return (value + tangent).detach()


class LearnedMapStep:
    def __call__(self, predictor, state, source, destination):
        return linear_transport(
            state, predictor(state, source, destination).exp(), source, destination
        )


class EndpointRefinementStep:
    """Pure endpoint composition T^k; no confidence-based renoising."""

    def __call__(self, predictor, state, source, destination):
        return predictor(state).exp()


def iterate_maps(predictor, state, times, policy, prompt_state=None, prompt_mask=None):
    """Shared differentiable iteration; inference callers choose no_grad.

    Every step costs one network evaluation. No extra terminal prediction is
    required: argmax the final state. Prompt clamping is performed every step.
    """
    times = torch.as_tensor(times, device=state.device, dtype=state.dtype)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a vector with at least two entries")
    if (
        not torch.isfinite(times).all()
        or ((times < 0) | (times > 1)).any()
        or (times[1:] > times[:-1]).any()
    ):
        raise ValueError("times must be finite, decreasing and within [0, 1]")
    if (prompt_state is None) != (prompt_mask is None):
        raise ValueError("prompt_state and prompt_mask must be supplied together")

    def clamp(value):
        return (
            value
            if prompt_mask is None
            else torch.where(prompt_mask[..., None].bool(), prompt_state, value)
        )

    state = clamp(state)
    for source, destination in zip(times[:-1], times[1:]):
        state = clamp(
            policy(
                predictor,
                state,
                source.expand(len(state)),
                destination.expand(len(state)),
            )
        )
    return state, len(times) - 1
