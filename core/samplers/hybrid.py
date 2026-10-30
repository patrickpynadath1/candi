"""Backward-compatible native-update preset for the shared sampler."""

from .composed import ComposedSampler
from .policies import TokenSelection, ReverseTransition


class HybridSampler(ComposedSampler):
    def __init__(
        self,
        steps=180,
        epsilon=1e-5,
        final_denoise="greedy",
        final_transition_noise=False,
        synchronize_committed=False,
        greedy_clean_estimate=False,
    ):
        self.greedy_clean_estimate = bool(greedy_clean_estimate)
        self.final_transition_noise = bool(final_transition_noise)
        super().__init__(
            steps,
            epsilon,
            tokens=TokenSelection("greedy" if greedy_clean_estimate else "posterior"),
            transition=ReverseTransition(
                clean_estimate=greedy_clean_estimate,
                final_transition_noise=final_transition_noise,
            ),
            final_denoise=final_denoise,
            final_prediction="greedy" if final_denoise == "greedy" else "sampling",
            synchronize_committed=synchronize_committed,
        )
