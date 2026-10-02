import torch
from .discrete import DiscreteSampler

from .utils import initialize_state


class AbsorbingAncestralUpdate:
    prediction_feedback = True

    def __init__(self, steps=180, epsilon=1e-5, final_denoise="greedy"):
        if steps <= 0 or not 0 < epsilon < 1:
            raise ValueError("positive steps and epsilon in (0, 1) required")
        if final_denoise not in {"greedy", "ancestral", "none"}:
            raise ValueError("unknown final denoising policy")
        self.steps, self.epsilon, self.final_denoise = steps, epsilon, final_denoise

    def reverse_step(self, algo, state, source_time, destination_time):
        keep_t, _ = algo.schedule(source_time.expand(len(state)))
        keep_s, _ = algo.schedule(destination_time.expand(len(state)))
        posterior = algo.sampling_prediction(state, keep_t).double().exp()
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
        update_rule=None,
        time_grid=None,
    ):
        rule = update_rule or AbsorbingAncestralUpdate(steps, epsilon, final_denoise)
        super().__init__(rule, steps, epsilon, time_grid)
        self.final_denoise = final_denoise

    def reverse_step(self, algo, state, source_time, destination_time):
        return self.update_rule.reverse_step(algo, state, source_time, destination_time)

    def ancestral_denoise(self, algo, state, time):
        return self.update_rule.ancestral_denoise(algo, state, time)

    def greedy_denoise(self, algo, state, time):
        return self.update_rule.greedy_denoise(algo, state, time)
