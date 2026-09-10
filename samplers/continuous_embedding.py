from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import run_continuous_diffusion
from .utils import continuous_time_grid, preserve_prompt
import torch
from algos.algo_utils import sample_categorical


class ContinuousEmbeddingSampler(Sampler):
    spec = SamplerSpec(
        name="continuous_embedding",
        compatible_families=frozenset({"continuous_embedding"}),
        required_capabilities=frozenset(
            {"continuous_prior", "continuous_schedule", "continuous_transition"}
        ),
    )

    def sample(self, model, request):
        samples, nfe = run_continuous_diffusion(model, request)
        return SamplingResult(samples, {"nfe": nfe})


@sampler_registry.register("continuous_embedding")
def _create(_config):
    return ContinuousEmbeddingSampler()


class CANDIEmbeddingSampler(Sampler):
    spec = SamplerSpec(
        name="candi_embedding", compatible_families=frozenset({"candi_embedding"}),
        required_capabilities=frozenset({"continuous_prior", "continuous_schedule", "continuous_transition"}),
    )

    @torch.no_grad()
    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        rho = 1-model.noise.alpha_t(times)
        u = model.sampling_continuous_schedule(times)
        state = model.prior_sample(request.num_samples, model.num_tokens)
        tokens = torch.zeros(state.shape[:2], dtype=torch.long, device=state.device)
        clean = torch.zeros_like(tokens, dtype=torch.bool)
        if request.prompt_tokens is not None:
            tokens = request.prompt_tokens.clone()
            clean = request.prompt_mask.bool().clone()
        state = model.sampling_prompt_state(state, tokens, clean)
        probs = None
        for i in range(steps):
            state, probs = model.sample_continuous_transition(
                state, rho[i], rho[i+1], sigma_t=u[i], sigma_s=u[i+1],
                clean_mask=clean, clean_tokens=tokens,
            )
            if model.hybrid_mask:
                probability = ((rho[i]-rho[i+1])/rho[i]).clamp(0, 1)
                reveal = (~clean) & (torch.rand_like(tokens, dtype=torch.float32) < probability)
                prediction = sample_categorical(probs)
                tokens = torch.where(reveal, prediction, tokens)
                clean |= reveal
            state = model.sampling_prompt_state(state, tokens, clean)
        if not model.hybrid_mask:
            tokens = probs.argmax(-1)
        return SamplingResult(preserve_prompt(tokens, request), {"nfe": steps})


@sampler_registry.register("candi_embedding")
def _create_candi_embedding(_config):
    return CANDIEmbeddingSampler()
