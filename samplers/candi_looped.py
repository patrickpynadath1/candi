"""Standard CANDI transitions with loop-aware model evaluation accounting."""

from .base import SamplerSpec
from .candi import CANDISampler
from .candi_entropy_bounded import CANDIEntropyBoundedSampler
from .registry import sampler_registry


class RecurrentCANDIMixin:
    def _transition_kwargs(self, model):
        # A fresh closure per sampling request keeps carry out of model state.
        hidden = None

        def denoise(**kwargs):
            nonlocal hidden
            log_probs, hidden = model._forward_once(kwargs, hidden)
            hidden = hidden.detach()
            return log_probs

        return {"denoise_fn": denoise}


class LoopedCANDISampler(RecurrentCANDIMixin, CANDISampler):
    spec = SamplerSpec(
        name="candi_looped",
        compatible_families=frozenset({"candi_looped"}),
        required_capabilities=CANDISampler.spec.required_capabilities,
    )

    def sample(self, model, request):
        result = super().sample(model, request)
        steps = result.metadata["nfe"]
        result.metadata.update(
            diffusion_steps=steps, recurrence="previous_step"
        )
        return result


@sampler_registry.register("candi_looped")
def _create(config):
    return LoopedCANDISampler(
        getattr(config.sampling, "trace_argmax_changes", False),
        getattr(config.sampling, "greedy_clean_estimate", False),
    )


class LoopedCANDIEntropyBoundedSampler(
    RecurrentCANDIMixin, CANDIEntropyBoundedSampler
):
    spec = SamplerSpec(
        name="candi_looped_entropy_bounded",
        compatible_families=frozenset({"candi_looped"}),
        required_capabilities=CANDIEntropyBoundedSampler.spec.required_capabilities,
    )

    def sample(self, model, request):
        result = super().sample(model, request)
        steps = result.metadata["nfe"]
        per_sample = result.metadata["nfe_per_sample"]
        result.metadata.update(
            diffusion_steps=steps,
            diffusion_steps_per_sample=per_sample,
            recurrence="previous_step",
        )
        return result


@sampler_registry.register("candi_looped_entropy_bounded")
def _create_entropy_bounded(config):
    return LoopedCANDIEntropyBoundedSampler(
        getattr(config.sampling, "entropy_bound", 0.0),
        getattr(config.sampling, "greedy_clean_estimate", False),
        getattr(config.sampling, "entropy_untempered", False),
    )
