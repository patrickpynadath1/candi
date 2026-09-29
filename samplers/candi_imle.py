"""Target-free, single-pass generation for conditional IMLE-style training."""

from .base import Sampler, SamplerSpec, SamplingResult
from .entropy_utils import sample_probabilities
from .registry import sampler_registry
from .utils import inject_onehot_prompt, preserve_prompt


class CANDIIMLESampler(Sampler):
    spec = SamplerSpec(
        name="candi_imle",
        compatible_families=frozenset({"candi_imle"}),
        required_capabilities=frozenset({"continuous_prior", "imle_prediction"}),
    )

    def __init__(self, greedy_clean_estimate=True):
        self.greedy_clean_estimate = bool(greedy_clean_estimate)

    def sample(self, model, request):
        state, reveal = inject_onehot_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        log_probs = model.sample_imle_distribution(state, reveal)
        tokens = (
            log_probs.argmax(-1) if self.greedy_clean_estimate
            else sample_probabilities(log_probs.exp())
        )
        return SamplingResult(preserve_prompt(tokens, request), {
            "nfe": 1,
            "nfe_per_sample": [1] * request.num_samples,
            "greedy_clean_estimate": self.greedy_clean_estimate,
            "training_candidates": model.config.algo.imle_candidates,
        })


@sampler_registry.register("candi_imle")
def _create(config):
    return CANDIIMLESampler(config.sampling.greedy_clean_estimate)
