"""Entropy-bounded commitment (https://arxiv.org/abs/2505.24857) on the shared loop.

Each step commits the lowest-entropy generated positions whose summed entropy stays
within ``entropy_bound``. Everything else follows each view's own reverse trajectory,
for any process: the sampler has no per-process code.

- ``freeze=True``: committed tokens stay fixed (optionally reopened by ``churn``).
- ``freeze=False``: every step re-selects; positions not selected again are re-noised
  through their view's forward kernel, and a row ends when all of it is selected at once.
"""

from .composed import ComposedSampler
from .policies import TokenSelection, EntropyPositions, ReverseTransition, select_prefix  # noqa: F401  (select_prefix re-exported)


class EntropyBoundedSampler(ComposedSampler):
    def __init__(
        self,
        *,
        entropy_bound=0.1,
        steps=127,
        epsilon=1e-5,
        freeze=True,
        churn=None,
        greedy_clean_estimate=True,
        continuous_noise="zero",
        precision="float32",
        require_completion=False,
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
            # continuous_noise: "zero" runs each continuous view's deterministic reverse
            # update, "native" its stochastic one; discrete views ignore it.
            transition=ReverseTransition(
                noise=continuous_noise, preserve_absorbing=True, renoise_released=not freeze
            ),
            commitment="freeze" if freeze else "revise",
            churn=churn,
            final_denoise="greedy" if greedy_clean_estimate else "sample",
            final_prediction="sampling",
            precision=precision,
            require_completion=require_completion,
        )

    def sample(self, algo, prompt_tokens, prompt_mask):
        output, info = super().sample(algo, prompt_tokens, prompt_mask)
        return output, {**info, "entropy_bound": self.bound, "freeze": self.freeze}
