import torch

from .utils import initialize_state
from core.modifiers.self_conditioning.session import prediction_session


class AncestralSampler:
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

    def sample(self, algo, prompt_tokens, prompt_mask):
        algo = prediction_session(algo)
        state = initialize_state(algo, prompt_tokens, prompt_mask)
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=state.device)
        # Matches the historical ancestral_cache trajectory: all requested
        # reverse steps, followed by a separate final denoising evaluation.
        for t, s in zip(times[:-1], times[1:]):
            state = self.reverse_step(algo, state, t, s)
        nfe = self.steps
        if self.final_denoise == "greedy":
            state = self.greedy_denoise(algo, state, times[-1])
            nfe += 1
        elif self.final_denoise == "ancestral":
            state = self.ancestral_denoise(algo, state, times[-1])
            nfe += 1
        return state, {"nfe": nfe, "nfe_per_sample": [nfe] * len(state)}
