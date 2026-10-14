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

    requires_clean_posterior = True

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

    requires_clean_posterior = False

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


class UniformInsertion(nn.Module):
    """Uniform noise at birth, then a birth-relative uniform replacement bridge.

    A position born at b has clean keep rate (alpha(t)-alpha(b))/(1-alpha(b)).
    It starts at zero signal at t=b and reaches the clean endpoint at t=0.
    Training and reverse transitions use this same conditional path.
    """

    requires_clean_posterior = False

    def validate(self, view):
        from core.processes.uniform_posterior import PosteriorUniformProcess

        if not isinstance(view.process, PosteriorUniformProcess):
            raise TypeError("UniformInsertion requires PosteriorUniformProcess")
        if hasattr(view.schedule, "for_layout"):
            raise TypeError("UniformInsertion requires a token-local schedule")

    def initialize(self, view, posterior, local_point):
        self.validate(view)
        return view.process.prior(posterior.shape[:2], posterior.device)

    def _keep(self, view, local_point, birth_times):
        born = (
            point(view, birth_times.flatten()).coefficients[0].reshape_as(birth_times)
        )
        current = local_point.coefficients[0][:, None]
        return ((current - born) / (1 - born).clamp_min(1e-7)).clamp(0, 1)

    def corrupt_training(self, view, tokens, local_point, birth_times, editable):
        keep = self._keep(view, local_point, birth_times)
        return view.process.corrupt(
            tokens.reshape(-1, 1), keep.flatten(), editable.reshape(-1, 1)
        ).reshape_as(tokens)

    def update(self, view, state, posterior, source, destination, birth_times):
        # The process consumes one keep rate per row. Flatten positions to use
        # its exact posterior bridge with each position's own birth time.
        return view.process.reverse(
            state.reshape(-1, 1),
            posterior.reshape(-1, 1, posterior.shape[-1]),
            self._keep(view, source, birth_times).flatten(),
            self._keep(view, destination, birth_times).flatten(),
        ).reshape_as(state)
