"""Algorithm 2: full simplex state, sampled clean vertices, optional churn."""

import torch
from algos.simplex_utils import reverse_bridge, guided_log_probs
from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


class SimplexSampler(Sampler):
    spec = SamplerSpec(
        "simplex",
        frozenset({"simplex"}),
        frozenset({"simplex_denoise", "simplex_prior", "simplex_concentration"}),
    )

    def __init__(self, config):
        self.churn = config.sampling.get("churn", 1.0)
        self.schedule = config.sampling.get("time_grid", "linear")
        self.guidance = config.sampling.get("guidance_scale")
        self.confidence_power = config.sampling.get("confidence_power")
        if self.guidance is not None and self.guidance < 0:
            raise ValueError("guidance_scale must be nonnegative")
        if self.confidence_power is not None and self.confidence_power <= 0:
            raise ValueError("confidence_power must be positive")
        if not 0 <= self.churn <= 1:
            raise ValueError("churn must be in [0,1]")

    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        times = model.paper_time_sampler.sampling_grid(
            steps, self.schedule, model.device
        ).double()
        state = model.prior_sample(request.num_samples, model.num_tokens)
        prompt = mask = None
        unconditional_carry = None
        controller = getattr(model.backbone, "latent_self_conditioning", None)
        if request.prompt_tokens is not None:
            prompt = (
                torch.nn.functional.one_hot(request.prompt_tokens, model.vocab_size)
                .to(state)
                .log()
            )
            mask = request.prompt_mask.bool()[..., None]
        for index in range(steps):
            if mask is not None:
                state = torch.where(mask, prompt, state)
            t = times[index].expand(request.num_samples)
            prediction = model(state, t.float()) / model.config.algo.temp
            if self.guidance is not None:
                if mask is None:
                    raise ValueError(
                        "classifier-free guidance requires a conditioning prompt"
                    )
                if model.config.algo.condition_dropout <= 0:
                    raise ValueError(
                        "classifier-free guidance requires condition-dropout training"
                    )
                conditional_carry = (
                    controller.previous if controller is not None else None
                )
                if controller is not None:
                    controller.previous = unconditional_carry
                unconditional = (
                    model(state.masked_fill(mask, -torch.inf), t.float())
                    / model.config.algo.temp
                )
                if controller is not None:
                    unconditional_carry = controller.previous
                    controller.previous = conditional_carry
                prediction = guided_log_probs(unconditional, prediction, self.guidance)
            tokens = torch.distributions.Categorical(logits=prediction).sample()
            if index == steps - 1:
                if self.confidence_power is not None:
                    tokens = prediction.argmax(-1)
                break
            s = times[index + 1].expand_as(t)
            state = reverse_bridge(
                state,
                tokens,
                s,
                t,
                model.concentration(s),
                model.concentration(t),
                model.reference.double(),
                self.churn,
            )
            # Appendix J.8 confidence commits: recompute the committed set on
            # every step; only generated positions contribute to its size.
            if self.confidence_power is not None:
                probabilities = prediction.softmax(-1)
                confidence, greedy = probabilities.max(-1)
                eligible = torch.ones_like(confidence, dtype=torch.bool)
                if mask is not None:
                    eligible &= ~mask.squeeze(-1)
                rank = (
                    confidence.masked_fill(~eligible, -torch.inf)
                    .argsort(-1, descending=True)
                    .argsort(-1)
                )
                count = (
                    (((1 - s) ** self.confidence_power) * eligible.sum(-1))
                    .floor()
                    .long()
                )
                committed = eligible & (rank < count[:, None])
                vertex = (
                    torch.nn.functional.one_hot(greedy, model.vocab_size)
                    .to(state)
                    .log()
                )
                state = torch.where(committed[..., None], vertex, state)
        return SamplingResult(
            preserve_prompt(tokens, request),
            {"nfe": steps * (2 if self.guidance is not None else 1)},
        )


@sampler_registry.register("simplex")
def _create(config):
    return SimplexSampler(config)
