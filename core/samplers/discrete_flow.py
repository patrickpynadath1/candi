"""DFM's sampled-endpoint CTMC update for discrete mixture paths.

The shared sampler selects an endpoint token. With clean mass k(t), decreasing
noise time t, and local step h=t-s, the conditional jump rate is
-k'(t)/(1-k(t)). A frozen-rate step jumps with probability 1-exp(-h*rate).
This is the basic (zero divergence-free term) reference DFM solver update;
it is not the uniform posterior ancestral bridge.
"""

import torch

from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform_posterior import PosteriorUniformProcess


class DiscreteFlowTransition:
    """Transition policy for ComposedSampler; token selection remains separate."""

    requires_tokens = True

    def validate(self, views, commitment):
        for view in views:
            if not isinstance(
                view.process, (AbsorbingProcess, PosteriorUniformProcess)
            ):
                raise TypeError(
                    "DFM requires absorbing or posterior-uniform mixture paths; "
                    "plug-in weights are not clean-token posteriors"
                )
            if getattr(view.schedule, "output_kind", None) != "keep_rate":
                raise TypeError("DFM requires a keep-rate schedule with its derivative")

    def __call__(
        self, view, state, probabilities, tokens, source, destination, **kwargs
    ):
        keep, derivative = source.coefficients
        next_keep, _ = destination.coefficients
        step = source.time - destination.time
        if (
            not torch.isfinite(keep).all()
            or not torch.isfinite(derivative).all()
            or not torch.isfinite(next_keep).all()
            or not torch.isfinite(step).all()
            or torch.any((keep < 0) | (next_keep > 1) | (next_keep < keep))
            or torch.any(derivative > 0)
            or torch.any(step < 0)
        ):
            raise ValueError("DFM requires finite, monotonically increasing clean mass")
        # At a clean endpoint there is no remaining source mass to transport.
        remaining = 1 - keep
        rate = torch.where(
            remaining > 0,
            -derivative / remaining.clamp_min(torch.finfo(keep.dtype).tiny),
            0,
        )
        jump_probability = -torch.expm1(-step * rate)
        jump = torch.rand(state.shape, device=state.device) < jump_probability[:, None]
        if isinstance(view.process, AbsorbingProcess):
            jump &= state.eq(view.process.mask_index)
        return torch.where(jump, tokens, state)
