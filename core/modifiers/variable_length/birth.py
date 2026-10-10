"""View-local insertion laws, independent of edit coordination.

Custom laws implement initialize(view, posterior, point) and optionally
corrupt_training(view, clean_tokens, point, birth_times, editable). Initializers
receive one posterior per newly created position, allowing independent draws
when several positions are inserted into the same gap.
"""

import torch
from torch import nn
from core.modifiers.hybrid.views import point


class ForwardInsertion(nn.Module):
    """Estimate x0, then apply the view's forward kernel at its CURRENT point."""

    def __init__(self, estimate="sample"):
        super().__init__()
        if estimate not in {"sample", "mean"}:
            raise ValueError("insertion estimate must be sample or mean")
        self.estimate = estimate

    def validate(self, view):
        if self.estimate == "mean" and not callable(
            getattr(view.representation, "posterior_mean", None)
        ):
            raise TypeError("mean insertion requires representation.posterior_mean")
        if hasattr(view.schedule, "for_layout"):
            raise TypeError(
                "sequence-coupled corruption needs a layout-aware insertion law"
            )

    def initialize(self, view, posterior, local_point):
        self.validate(view)
        if self.estimate == "mean":
            clean = view.representation.posterior_mean(posterior)
        else:
            tokens = torch.distributions.Categorical(probs=posterior).sample()
            clean = view.representation.encode(tokens)
        active = torch.ones(
            posterior.shape[:2], dtype=torch.bool, device=posterior.device
        )
        return view.process.corrupt_state(clean, local_point.coefficients, active)

    def corrupt_training(self, view, tokens, local_point, birth_times, editable):
        del birth_times
        return view.process.corrupt_state(
            view.representation.encode(tokens), local_point.coefficients, editable
        )


class MaskInsertion(nn.Module):
    """Absorbing masks at birth, followed by the view's unmasking hazard.

    Training conditions unmasking on birth having happened, rather than independently
    masking the compacted sequence. This is the FlexMDM coupling for a token-local
    absorbing view, including its configured global-to-local time map.
    """

    def validate(self, view):
        if not hasattr(view.process, "mask_index") or not callable(
            getattr(view.process, "observed_tokens", None)
        ):
            raise TypeError("MaskInsertion requires an absorbing process")
        if hasattr(view.schedule, "for_layout"):
            raise TypeError("MaskInsertion requires a token-local schedule")

    def initialize(self, view, posterior, local_point):
        self.validate(view)
        tokens = torch.zeros(
            posterior.shape[:2], dtype=torch.long, device=posterior.device
        )
        return view.process.prior_state(tokens, local_point.coefficients)

    def corrupt_training(self, view, tokens, local_point, birth_times, editable):
        keep_now = local_point.coefficients[0][:, None]
        if birth_times.numel() == 0:
            return tokens
        birth_keep = (
            point(view, birth_times.flatten()).coefficients[0].reshape_as(birth_times)
        )
        masked_probability = (1 - keep_now) / (1 - birth_keep).clamp_min(1e-7)
        if (masked_probability[editable] > 1 + 1e-5).any():
            raise ValueError(
                "unmasking schedule must be monotone along insertion trajectories"
            )
        masked = torch.rand(
            tokens.shape, device=tokens.device
        ) < masked_probability.clamp(0, 1)
        return torch.where(editable & masked, view.process.mask_index, tokens)
