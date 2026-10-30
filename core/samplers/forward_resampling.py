"""Forward-resampling preset for the shared fixed-length sampler."""

from .composed import ComposedSampler
from .policies import TokenSelection, ForwardCorruption


class ForwardResamplingSampler(ComposedSampler):
    def __init__(
        self,
        steps=180,
        epsilon=1e-5,
        greedy_clean_estimate=False,
        final_denoise="greedy",
    ):
        self.greedy_clean_estimate = bool(greedy_clean_estimate)
        super().__init__(
            steps,
            epsilon,
            tokens=TokenSelection("greedy" if greedy_clean_estimate else "sample"),
            transition=ForwardCorruption(),
            final_denoise=final_denoise,
            final_prediction="greedy" if final_denoise == "greedy" else "sampling",
        )
