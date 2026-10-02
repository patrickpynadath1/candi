"""Clean-token posterior pathways and their owned inference configuration."""

import math
import torch


from .utils import (
    softcap_logits,
    validate_softcap,
    scale_temperature,
    filter_top_k,
    filter_top_p,
    greedy_projection,
    greedy_probabilities,
    normalize_log_probs,
    predictive_entropy,
)

from .constraints import AbsorbingConstraints, UnconstrainedPosterior


class PosteriorParameterization:
    def __init__(
        self,
        mask_index=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        commitment="greedy",
        constraints=None,
        softcap=None,
    ):
        validate_softcap(softcap)
        self.softcap = softcap
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not isinstance(top_k, int) or top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("top_k must be a nonnegative integer and top_p in (0, 1]")
        if commitment not in {"greedy", "sample"}:
            raise ValueError("commitment must be greedy or sample")
        self.mask_index = mask_index
        self.constraints = (
            constraints
            if constraints is not None
            else (
                UnconstrainedPosterior()
                if mask_index is None
                else AbsorbingConstraints(mask_index)
            )
        )
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.commitment = commitment
        self._probabilities = {
            "greedy": greedy_probabilities,
            "sample": torch.exp,
        }[commitment]

    def _constrain(self, values, observed):
        return self.constraints.condition(normalize_log_probs(values), observed)

    def _shape(self, values):
        values = scale_temperature(values, self.temperature)
        values = filter_top_k(values, self.top_k)
        return filter_top_p(values, self.top_p)

    def training_posterior(self, logits, observed):
        """Model softcapping and constraints, without inference-only shaping."""
        return self._constrain(self._model_values(logits), observed)

    def _model_values(self, logits):
        return self.constraints.exclude(softcap_logits(logits, self.softcap))

    def sampling_posterior(self, logits, observed):
        """Configured shaping, normalized at the baseline inference precision."""
        with torch.autocast(logits.device.type, enabled=False):
            values = self._shape(self._model_values(logits))
            return self._constrain(values, observed)

    def greedy_posterior(self, logits, observed):
        """Explicit greedy denoising, independent of the EB commitment policy."""
        with torch.autocast(logits.device.type, enabled=False):
            values = self._shape(self._model_values(logits))
            return self._constrain(greedy_projection(values), observed)

    def entropy(self, log_probs):
        """Measure the full posterior before commitment projection."""
        return predictive_entropy(log_probs)

    def probabilities(
        self,
        prediction,
        *,
        apply_commitment=False,
    ):
        """Convert an already constrained prediction, preserving its precision.

        Optionally apply the configured commitment policy. Entropy should be
        measured on the original prediction, before this projection.
        """
        if apply_commitment:
            return self._probabilities(prediction)
        return prediction.exp()
