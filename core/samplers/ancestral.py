import torch
from .discrete import DiscreteSampler

from .utils import initialize_state
from . import requirements


class AbsorbingAncestralUpdate:
    prediction_feedback = True

    def __init__(
        self,
        steps=180,
        epsilon=1e-5,
        final_denoise="greedy",
        greedy_clean_estimate=False,
    ):
        if steps <= 0 or not 0 < epsilon < 1:
            raise ValueError("positive steps and epsilon in (0, 1) required")
        if final_denoise not in {"greedy", "ancestral", "none"}:
            raise ValueError("unknown final denoising policy")
        self.steps, self.epsilon, self.final_denoise = steps, epsilon, final_denoise
        self.greedy_clean_estimate = bool(greedy_clean_estimate)

    def reverse_step(self, algo, state, source_time, destination_time):
        keep_t, _ = algo.schedule(source_time.expand(len(state)))
        keep_s, _ = algo.schedule(destination_time.expand(len(state)))
        predict = (
            algo.greedy_prediction
            if self.greedy_clean_estimate
            else algo.sampling_prediction
        )
        posterior = predict(state, keep_t).double().exp()
        return algo.process.reverse(state, posterior, keep_t, keep_s)

    def ancestral_denoise(self, algo, state, time):
        keep, _ = algo.schedule(time.expand(len(state)))
        posterior = algo.sampling_prediction(state, keep).double().exp()
        return algo.process.reverse(state, posterior, keep, torch.ones_like(keep))

    def greedy_denoise(self, algo, state, time):
        keep, _ = algo.schedule(time.expand(len(state)))
        posterior = algo.greedy_prediction(state, keep).double().exp()
        return algo.process.commit(
            state, posterior, state.eq(algo.process.mask_index), deterministic=True
        )

    def initialize(self, algo, prompt_tokens, prompt_mask):
        return initialize_state(algo, prompt_tokens, prompt_mask)

    def update(self, algo, state, time, next_time, prompt_tokens, prompt_mask):
        return self.reverse_step(algo, state, time, next_time), 1

    def finalize(self, algo, state, time, prompt_tokens, prompt_mask):
        if self.final_denoise == "greedy":
            return self.greedy_denoise(algo, state, time), 1
        if self.final_denoise == "ancestral":
            return self.ancestral_denoise(algo, state, time), 1
        return state, 0


class AncestralSampler(DiscreteSampler):
    def __init__(
        self,
        steps=180,
        epsilon=1e-5,
        final_denoise="greedy",
        *,
        greedy_clean_estimate=False,
        update_rule=None,
        time_grid=None,
    ):
        rule = update_rule or AbsorbingAncestralUpdate(
            steps, epsilon, final_denoise, greedy_clean_estimate
        )
        super().__init__(rule, steps, epsilon, time_grid)
        self.final_denoise = final_denoise
        self.greedy_clean_estimate = bool(greedy_clean_estimate)
        self._custom = update_rule is not None or time_grid is not None

    def composed(self):
        """The shared-loop preset that is bit-identical to this sampler."""
        clean = "greedy" if self.greedy_clean_estimate else "sampling"
        return self._composed("float64", clean)

    def sample(self, algo, prompt_tokens, prompt_mask):
        if hasattr(algo, "views"):
            return self.composed().sample(algo, prompt_tokens, prompt_mask)
        requirements.process_method(algo, "reverse", "ancestral sampling", "discrete")
        return super().sample(algo, prompt_tokens, prompt_mask)
