"""Entropy-selected commitment preset for the shared fixed-length sampler."""

from .composed import ComposedSampler
from .policies import TokenSelection, EntropyPositions, ReverseTransition


class ParallelEntropyBoundedSampler(ComposedSampler):
    def __init__(
        self,
        *,
        freeze=True,
        entropy_bound=0.1,
        steps=180,
        epsilon=1e-5,
        greedy_clean_estimate=True,
    ):
        self.freeze = bool(freeze)
        self.bound = float(entropy_bound)
        self.greedy_clean_estimate = bool(greedy_clean_estimate)
        super().__init__(
            steps,
            epsilon,
            tokens=TokenSelection(
                "greedy" if greedy_clean_estimate else "sample",
                distribution="probabilities",
            ),
            positions=EntropyPositions(entropy_bound),
            transition=ReverseTransition(
                noise="zero", preserve_absorbing=True, renoise_released=not freeze
            ),
            commitment="freeze" if freeze else "revise",
            final_denoise="greedy" if greedy_clean_estimate else "sample",
            final_prediction="sampling",
        )

    def sample(self, algo, prompt_tokens, prompt_mask):
        output, info = super().sample(algo, prompt_tokens, prompt_mask)
        return output, {**info, "entropy_bound": self.bound, "freeze": self.freeze}
