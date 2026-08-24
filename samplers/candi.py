import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import continuous_time_grid, inject_onehot_prompt, preserve_prompt


class CANDISampler(Sampler):
    spec = SamplerSpec(
        name="candi",
        compatible_families=frozenset({"candi"}),
        required_capabilities=frozenset({
            "continuous_prior", "continuous_schedule", "candi_discrete_schedule",
            "candi_continuous_transition", "candi_discrete_transition",
        }),
    )

    def __init__(self, trace_argmax_changes=False):
        self.trace_argmax_changes = bool(trace_argmax_changes)

    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        x = model.prior_sample(request.num_samples, model.num_tokens)
        prompt_state_fn = getattr(model, "sampling_prompt_state", None)
        if callable(prompt_state_fn):
            x, clean = prompt_state_fn(
                x, request.prompt_tokens, request.prompt_mask
            )
        else:
            x, clean = inject_onehot_prompt(x, request)
        sigmas = model.sampling_continuous_schedule(times, steps)
        discrete_times = model.sampling_discrete_schedule(times)
        previous_argmax = None
        changes_per_sample = torch.zeros(
            x.shape[0], dtype=torch.long, device=x.device
        )
        changed_positions = torch.zeros(
            x.shape[:2], dtype=torch.bool, device=x.device
        )
        changes_per_step = []
        for i in range(steps):
            continuous, clean_probs = model.sample_candi_continuous_transition(
                x, discrete_times[i], sigma_s=sigmas[i], sigma_t=sigmas[i + 1],
                clean_mask=clean, time_s=discrete_times[i + 1],
            )
            if self.trace_argmax_changes:
                current_argmax = clean_probs.argmax(dim=-1)
                if previous_argmax is not None:
                    changed = current_argmax.ne(previous_argmax)
                    if request.prompt_mask is not None:
                        changed &= ~request.prompt_mask.bool()
                    step_counts = changed.sum(dim=-1)
                    changes_per_sample += step_counts
                    changed_positions |= changed
                    changes_per_step.append(step_counts.cpu().tolist())
                previous_argmax = current_argmax
            x, clean = model.sample_candi_discrete_transition(
                continuous,
                clean_probs,
                discrete_times[i],
                discrete_times[i] - discrete_times[i + 1],
                prev_clean_mask=clean,
            )
        state_tokens_fn = getattr(model, "sampling_state_tokens", None)
        tokens = state_tokens_fn(x) if callable(state_tokens_fn) else x.argmax(-1)
        metadata = {"nfe": steps}
        if self.trace_argmax_changes:
            metadata.update({
                "argmax_changes_per_sample": changes_per_sample.cpu().tolist(),
                "argmax_changed_positions_per_sample": (
                    changed_positions.sum(dim=-1).cpu().tolist()
                ),
                "argmax_changes_per_step": changes_per_step,
                "argmax_comparisons": max(0, steps - 1),
            })
        return SamplingResult(preserve_prompt(tokens, request), metadata)


@sampler_registry.register("candi")
def _create(config):
    return CANDISampler(
        getattr(config.sampling, "trace_argmax_changes", False)
    )
