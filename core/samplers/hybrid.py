"""Joint reverse sampler for independent parallel branch states."""

import torch
from core.modifiers.self_conditioning.session import prediction_session


class HybridSampler:
    def __init__(self, steps=180, epsilon=1e-5, final_denoise="greedy",
                 final_transition_noise=False, synchronize_committed=False):
        if steps < 1 or not 0 <= epsilon < 1:
            raise ValueError("positive steps and epsilon in [0, 1) required")
        if final_denoise not in {"greedy", "sample"}:
            raise ValueError("final_denoise must be greedy or sample")
        self.steps, self.epsilon = int(steps), float(epsilon)
        self.final_denoise = final_denoise
        self.final_transition_noise = bool(final_transition_noise)
        self.synchronize_committed = bool(synchronize_committed)

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        algo = prediction_session(algo)
        batch = len(prompt_tokens)
        device = prompt_tokens.device
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=device)
        first = times[0].expand(batch)
        points = algo.points(first)
        states = tuple(
            branch.prior(prompt_tokens, point, prompt_mask)
            for branch, point in zip(algo.branches, points)
        )
        nfe = 0
        for index, (source_time, destination_time) in enumerate(
            zip(times[:-1], times[1:])
        ):
            source_time = source_time.expand(batch)
            destination_time = destination_time.expand(batch)
            source, destination = algo.points(source_time), algo.points(destination_time)
            posterior = algo.sampling_prediction(
                states, source, source_time
            ).float().exp()
            nfe += 1
            updated = []
            for branch, state, point_t, point_s in zip(
                algo.branches, states, source, destination
            ):
                noise = (
                    torch.zeros_like(state) if state.is_floating_point()
                    and index == self.steps - 1 and not self.final_transition_noise
                    else None
                )
                value = branch.update(
                    state, posterior, point_t, point_s, noise=noise
                )
                updated.append(branch.clamp_prompt(
                    value, prompt_tokens, prompt_mask, point_s
                ))
            states = tuple(updated)
            if self.synchronize_committed:
                states = algo._synchronize_committed(states)
        final_time = times[-1].expand(batch)
        final_points = algo.points(final_time)
        prediction = (
            algo.greedy_prediction(states, final_points, final_time)
            if self.final_denoise == "greedy"
            else algo.sampling_prediction(states, final_points, final_time)
        )
        nfe += 1
        tokens = prediction.argmax(-1) if self.final_denoise == "greedy" else torch.distributions.Categorical(logits=prediction).sample()
        tokens = torch.where(prompt_mask.bool(), prompt_tokens, tokens)
        return tokens, {"nfe": nfe, "nfe_per_sample": [nfe] * batch}
