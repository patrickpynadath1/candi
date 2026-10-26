"""Independent choices for clean tokens, accepted positions, and noisy updates."""

import math
import torch
from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform import UniformProcess
from .entropy_bounded import select_prefix


class TokenSelection:
    def __init__(self, mode="sample", distribution="logits"):
        if mode not in {"posterior", "sample", "greedy"}:
            raise ValueError("token mode must be posterior, sample, or greedy")
        if distribution not in {"logits", "probabilities"}:
            raise ValueError("distribution must be logits or probabilities")
        self.mode, self.distribution = mode, distribution

    def __call__(self, logits, probabilities):
        if self.mode == "posterior":
            return None
        if self.mode == "greedy":
            return probabilities.argmax(-1)
        distribution = (
            torch.distributions.Categorical(logits=logits)
            if self.distribution == "logits"
            else torch.distributions.Categorical(probs=probabilities)
        )
        return distribution.sample()


class AllPositions:
    def __call__(self, probabilities, active):
        return active.clone()


class RandomPositions:
    """Accept up to count uniformly sampled active positions per example."""

    def __init__(self, count=1):
        if int(count) != count or count < 1:
            raise ValueError("random position count must be a positive integer")
        self.count = int(count)

    def __call__(self, probabilities, active):
        selected = torch.zeros_like(active)
        for row in range(len(active)):
            positions = active[row].nonzero(as_tuple=True)[0]
            order = torch.randperm(len(positions), device=active.device)
            selected[row, positions[order[: self.count]]] = True
        return selected


class EntropyPositions:
    def __init__(self, bound=0.1):
        if not math.isfinite(bound) or bound < 0:
            raise ValueError("entropy bound must be finite and nonnegative")
        self.bound = float(bound)

    def __call__(self, probabilities, active):
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        return select_prefix(entropy, active, self.bound)


def select_state(mask, yes, no):
    while mask.ndim < yes.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, yes, no)


class ReverseTransition:
    """Use each view's configured reverse rule, with explicit noise choices."""

    requires_tokens = False

    def __init__(
        self,
        clean_estimate=False,
        noise="native",
        final_transition_noise=False,
        preserve_absorbing=False,
        renoise_released=False,
    ):
        if noise not in {"native", "zero"}:
            raise ValueError("reverse noise must be native or zero")
        self.clean_estimate = bool(clean_estimate)
        self.requires_tokens = self.clean_estimate
        self.noise = noise
        self.final_transition_noise = bool(final_transition_noise)
        self.preserve_absorbing = bool(preserve_absorbing)
        self.renoise_released = bool(renoise_released)

    def validate(self, views, commitment):
        if (
            self.preserve_absorbing
            and commitment == "revise"
            and any(isinstance(v.process, AbsorbingProcess) for v in views)
        ):
            raise ValueError(
                "Revisable EB cannot bypass absorbing posterior constraints"
            )

    def __call__(
        self,
        view,
        state,
        probabilities,
        tokens,
        source,
        destination,
        *,
        final_step,
        generated,
        prompt_tokens,
        released,
        previous_tokens,
    ):
        if self.clean_estimate:
            probabilities = torch.nn.functional.one_hot(
                tokens, probabilities.shape[-1]
            ).to(probabilities)
        noise = (
            torch.zeros_like(state)
            if self.noise == "zero"
            or (
                state.is_floating_point()
                and final_step
                and not self.final_transition_noise
            )
            else None
        )
        value = (
            state
            if self.preserve_absorbing and isinstance(view.process, AbsorbingProcess)
            else view.update(
                view.process,
                view.representation,
                state,
                probabilities,
                source,
                destination,
                noise=noise,
            )
        )
        if (
            self.renoise_released
            and not isinstance(view.process, UniformProcess)
            and released.any()
        ):
            noisy = view.process.corrupt_state(
                view.representation.encode(previous_tokens),
                destination.coefficients,
                released,
            )
            value = select_state(released, noisy, value)
        return value


class ForwardCorruption:
    """Draw each view's forward kernel from the shared sampled clean sequence."""

    requires_tokens = True

    def validate(self, views, commitment):
        for view in views:
            if not callable(getattr(view.process, "corrupt_state", None)):
                raise TypeError("each view must expose a forward corrupt_state kernel")

    def __call__(
        self,
        view,
        state,
        probabilities,
        tokens,
        source,
        destination,
        *,
        final_step,
        generated,
        prompt_tokens,
        released,
        previous_tokens,
    ):
        clean = torch.where(generated, tokens, prompt_tokens)
        return view.process.corrupt_state(
            view.representation.encode(clean), destination.coefficients, generated
        )
