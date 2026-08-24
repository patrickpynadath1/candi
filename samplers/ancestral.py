import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import (
    apply_token_offset,
    diffusion_time_grid,
    inject_token_prompt,
    preserve_prompt,
)


class AncestralSampler(Sampler):
    def __init__(self, noise_removal, trace_argmax_changes=False):
        self.noise_removal = noise_removal
        self.trace_argmax_changes = bool(trace_argmax_changes)
        required = {"discrete_prior", "noise_schedule", "ancestral_transition"}
        if noise_removal == "greedy":
            required.add("greedy_denoise")
        self.spec = SamplerSpec(
            name="ancestral",
            compatible_families=frozenset(
                {"absorbing_diffusion", "uniform_diffusion"}
            ),
            required_capabilities=frozenset(required),
        )

    def sample(self, model, request):
        steps, times, dt = diffusion_time_grid(model, request)
        x = inject_token_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        previous_argmax = None
        changes_per_sample = torch.zeros(
            x.shape[0], dtype=torch.long, device=x.device
        )
        changed_positions = torch.zeros_like(x, dtype=torch.bool)
        changes_per_step = []

        def trace(prediction):
            nonlocal previous_argmax, changes_per_sample, changed_positions
            if not self.trace_argmax_changes or prediction is None:
                return
            current_argmax = prediction.argmax(dim=-1)
            if previous_argmax is not None:
                changed = current_argmax.ne(previous_argmax)
                if request.prompt_mask is not None:
                    changed &= ~request.prompt_mask.bool()
                step_counts = changed.sum(dim=-1)
                changes_per_sample += step_counts
                changed_positions |= changed
                changes_per_step.append(step_counts.cpu().tolist())
            previous_argmax = current_argmax

        for i in range(steps):
            t = times[i] * torch.ones(x.shape[0], 1, device=model.device)
            is_last = i == steps - 1
            if is_last and self.noise_removal == "greedy":
                x = model.sample_greedy_denoise(x=x, t=t)
                x = preserve_prompt(x, request)
            else:
                prediction, x = model.sample_ancestral_transition(
                    x=x,
                    t=t,
                    dt=dt,
                    p_x0=None,
                    noise_removal_step=(
                        is_last and self.noise_removal == "ancestral"
                    ),
                    prompt_mask=request.prompt_mask,
                )
                trace(prediction)
        nfe = steps
        metadata = {"nfe": nfe}
        if self.trace_argmax_changes:
            metadata.update({
                "argmax_changes_per_sample": changes_per_sample.cpu().tolist(),
                "argmax_changed_positions_per_sample": (
                    changed_positions.sum(dim=-1).cpu().tolist()
                ),
                "argmax_changes_per_step": changes_per_step,
                "argmax_comparisons": max(0, nfe - 1),
            })
        return SamplingResult(apply_token_offset(model, x), metadata)


@sampler_registry.register("ancestral")
def _create(config):
    return AncestralSampler(
        config.sampling.noise_removal,
        getattr(config.sampling, "trace_argmax_changes", False),
    )
