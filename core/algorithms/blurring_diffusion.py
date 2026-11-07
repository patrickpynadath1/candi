"""Separable positional/feature blurring diffusion as a one-view recipe.

Runs on the view path: one blurring view, held on the algorithm, over a
ModularDiT adopted from the configured ContinuousDiT (see
core/models/legacy.py:adopt). Training and sampling behave as the former
standalone implementation, which core/tests/reference_algorithms keeps for
comparison.
"""

from core.models.legacy import adopt
from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView
from core.samplers.hybrid_updates import BlurringPosteriorUpdate

# HybridDiffusion's per-call overrides; blurring never supported them.
_UNSUPPORTED_LOSS_OPTIONS = ("time_sampler", "objective", "pure_noise")


class BlurringDiffusion(OneViewAlgorithm, HybridDiffusion):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        representation,
        length=180,
        prediction=None,
        detach_forward_embedding=False,
    ):
        if getattr(objective, "requires_context", False):
            raise TypeError(
                "blurring diffusion supports token cross entropy without scalar Gaussian weighting"
            )
        if model.state_dim != representation.spec.dimension:
            raise ValueError("model state_dim must match blur representation dimension")
        if model.vocab_size != representation.spec.vocab_size:
            raise ValueError("model and representation vocabularies must match")
        if model.input_preconditioning != "none" or model.time_conditioning != "time":
            raise ValueError(
                "blurring diffusion requires raw time and no input preconditioning"
            )
        view = DiffusionView(process, schedule, representation, BlurringPosteriorUpdate())
        super().__init__(
            adopt(model),
            predictor,
            time_sampler,
            objective,
            length=length,
            prediction=prediction,
            views=[view],
        )
        self.detach_forward_embedding = bool(detach_forward_embedding)

    def prediction_mean(self, probabilities):
        # Self-conditioning feeds back the expected codebook vector (state space).
        return self.representation.posterior_mean(probabilities)

    def _corrupt(self, tokens, points, corruption_mask):
        # The schedule's layout path already restricts corruption to active spans.
        clean = self.representation.encode(tokens)
        if self.detach_forward_embedding:
            clean = clean.detach()  # Only the forward process input.
        return (self.process.corrupt(clean, points[0].coefficients),)

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        sequence_layout=None,
        step=0,
        training=False,
        **time_context,
    ):
        unsupported = sorted(set(time_context) & set(_UNSUPPORTED_LOSS_OPTIONS))
        if unsupported:
            raise TypeError(f"blurring diffusion does not support loss overrides: {unsupported}")
        return super().loss(
            tokens,
            loss_mask,
            corruption_mask,
            sequence_layout=sequence_layout,
            step=step,
            training=training,
            **time_context,
        )
