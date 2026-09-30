"""Optional latent self-conditioning shared by denoising backbones (paper I.3).

The adapter leaves parameter paths intact and brackets each denoiser call. The
carry exists only inside a sampling context; unrelated minibatches never share it.
"""

from contextlib import contextmanager

import torch
from torch import nn


class LatentSelfConditioning(nn.Module):
    def __init__(self, hidden_size, probability=0.9):
        super().__init__()
        if not 0 <= probability <= 1:
            raise ValueError("self-conditioning probability must be in [0,1]")
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.zeros_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        self.probability = probability
        self.previous = None
        self.current = None
        self.active = False
        self.sampling = False
        self.causal_prefix = False

    def inject(self, hidden):
        if not self.active:
            return hidden
        previous = self.previous
        if previous is None:
            previous = torch.zeros_like(hidden)
        if previous.shape != hidden.shape:
            if (
                self.causal_prefix
                and previous.shape[0] == hidden.shape[0]
                and previous.shape[2] == hidden.shape[2]
                and previous.shape[1] < hidden.shape[1]
            ):
                previous = torch.nn.functional.pad(
                    previous, (0, 0, 0, hidden.shape[1] - previous.shape[1])
                )
            else:
                raise ValueError("self-conditioning requires aligned latent positions")
        return hidden + self.norm(previous.detach().to(hidden)).to(hidden)

    @contextmanager
    def trajectory(self):
        old = self.sampling, self.previous, self.current
        self.sampling, self.previous, self.current = True, None, None
        try:
            yield
        finally:
            self.sampling, self.previous, self.current = old

    def run(self, function, *args, **kwargs):
        if self.active:
            return function(*args, **kwargs)
        # A mutable KV cache cannot be replayed for the detached pass, and a
        # changing prefix cannot reuse a position-aligned latent implicitly.
        if kwargs.get("cache") is not None:
            raise ValueError(
                "latent self-conditioning requires uncached full-sequence inference"
            )
        self.active = True
        try:
            if not self.sampling:
                self.previous = None
                if torch.rand(()).item() < self.probability:
                    with torch.no_grad():
                        function(*args, **kwargs)
                    if self.current is None:
                        raise RuntimeError("denoiser did not expose its final latent")
                    self.previous = self.current.detach()
            result = function(*args, **kwargs)
            if self.current is None:
                raise RuntimeError("denoiser did not expose its final latent")
            self.previous = self.current.detach() if self.sampling else None
            return result
        finally:
            self.active = False
            self.current = None
            if not self.sampling:
                self.previous = None


class _BlockCall:
    def __init__(self, function, condition, inject, capture):
        self.function, self.condition = function, condition
        self.inject, self.capture = inject, capture

    def __call__(self, hidden, *args, **kwargs):
        if self.inject:
            hidden = self.condition.inject(hidden)
        result = self.function(hidden, *args, **kwargs)
        if self.capture and self.condition.active:
            self.condition.current = result
        return result


class _EntryCall:
    def __init__(self, function, condition):
        self.function, self.condition = function, condition

    def __call__(self, *args, **kwargs):
        return self.condition.run(self.function, *args, **kwargs)


def install_self_conditioning(backbone, hidden_size, probability):
    """Install one adapter without changing existing parameter/checkpoint paths.

    Supported backbones expose a nonempty ``blocks`` sequence. Entry wrappers
    bracket an entire denoiser call, while block wrappers inject before the first
    decoder block and capture the last block output. Callable objects, rather
    than closures or bound hooks, keep deepcopy teachers independently wired.
    ``trajectory()`` is owned by the sampler runner, not individual samplers.
    """
    if hasattr(backbone, "latent_self_conditioning"):
        raise ValueError("latent self-conditioning is already installed")
    if not hasattr(backbone, "blocks") or not len(backbone.blocks):
        raise ValueError("latent self-conditioning requires a transformer block stack")
    if hasattr(backbone, "sequence_up"):
        hidden_size = backbone.sequence_up.out_features
    elif hasattr(backbone, "embed_dim") and not hasattr(backbone, "output_layer"):
        hidden_size = backbone.embed_dim
    condition = LatentSelfConditioning(hidden_size, probability)
    condition.causal_prefix = bool(getattr(backbone, "causal", False))
    backbone.latent_self_conditioning = condition
    first = getattr(backbone, "encoder_blocks", 0)

    for index in {first, len(backbone.blocks) - 1}:
        block = backbone.blocks[index]
        for name in ("forward", "forward_masked", "forward_positioned"):
            if hasattr(block, name):
                setattr(
                    block,
                    name,
                    _BlockCall(
                        getattr(block, name),
                        condition,
                        index == first,
                        index == len(backbone.blocks) - 1,
                    ),
                )

    names = (
        ("decode_latents",)
        if first
        else (
            "forward",
            "forward_with_embeddings",
            "forward_masked_embeddings",
            "forward_positioned",
        )
    )
    for name in names:
        if hasattr(backbone, name):
            setattr(backbone, name, _EntryCall(getattr(backbone, name), condition))
    return condition
