"""Normalized plug-in estimates for uniform reverse kernels, not posteriors."""

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
)


class PlugInParameterization:
    def __init__(
        self,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        commitment="sample",
        softcap=None,
    ):
        validate_softcap(softcap)
        self.softcap = softcap
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not isinstance(top_k, int) or top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid top-k/top-p")
        if commitment not in {"sample", "greedy"}:
            raise ValueError("commitment must be sample or greedy")
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.commitment = commitment
        self._probabilities = {
            "sample": torch.exp,
            "greedy": greedy_probabilities,
        }[commitment]

    def _shape(self, logits):
        values = scale_temperature(logits, self.temperature)
        return filter_top_p(filter_top_k(values, self.top_k), self.top_p)

    def training_plugin(self, logits):
        """Apply model softcapping, without inference-only shaping."""
        return softcap_logits(logits, self.softcap).log_softmax(-1)

    def sampling_plugin(self, logits):
        with torch.autocast(logits.device.type, enabled=False):
            return self._shape(softcap_logits(logits, self.softcap)).log_softmax(-1)

    def greedy_plugin(self, logits):
        with torch.autocast(logits.device.type, enabled=False):
            return greedy_projection(self._shape(softcap_logits(logits, self.softcap)))

    def entropy(self, prediction):
        probabilities = prediction.exp().float()
        return -torch.special.xlogy(probabilities, probabilities).sum(-1)

    def probabilities(
        self,
        prediction,
        *,
        apply_commitment=False,
    ):
        """Convert log probabilities, optionally applying the configured policy.

        Preserve input precision; callers choose precision for their operation.
        This returns a clean-token estimate, not a reverse transition.
        """
        if apply_commitment:
            return self._probabilities(prediction)
        return prediction.exp()
