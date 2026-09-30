"""Simplex diffusion: Dirichlet corruption, categorical CE denoising."""

import torch
import torch.nn.functional as F
from .diffusion import Diffusion
from .registry import algo_registry
from .simplex_utils import concentration, corruption_parameters, log_dirichlet_sample


@algo_registry.register("simplex")
class SimplexDiffusion(Diffusion):
    default_sampler = "simplex"
    supported_samplers = frozenset({"simplex"})
    sampling_family = "simplex"
    sampling_capabilities = frozenset(
        {"simplex_denoise", "simplex_prior", "simplex_concentration"}
    )

    def __init__(self, config, tokenizer):
        if config.algo.backbone != "dit" or config.algo.causal_attention:
            raise ValueError("Simplex diffusion requires a noncausal dit backbone")
        if config.noise.get("adaptive", False):
            raise ValueError(
                "Use time_sampling.adaptive for the paper's bucketed time sampler"
            )
        super().__init__(config, tokenizer, vocab_size=len(tokenizer))
        self.register_buffer(
            "reference", torch.full((self.vocab_size,), 1 / self.vocab_size)
        )
        reference = config.algo.get("reference")
        if reference is not None:
            reference = torch.as_tensor(list(reference), dtype=torch.float32)
            if (
                reference.shape != (self.vocab_size,)
                or not torch.isfinite(reference).all()
                or (reference < 0).any()
                or reference.sum() <= 0
            ):
                raise ValueError(
                    "reference must contain one finite nonnegative weight per token"
                )
            self.reference.copy_(reference / reference.sum())
        if not 0 <= config.algo.get("condition_dropout", 0.0) <= 1:
            raise ValueError("condition_dropout must be in [0,1]")
        if config.algo.input_mode not in {"expectation", "argmax"}:
            raise ValueError("simplex input_mode must be expectation or argmax")
        if config.algo.temp <= 0:
            raise ValueError("temperature must be positive")
        self.concentration(torch.ones(1))

    def concentration(self, t):
        a = self.config.algo
        return concentration(t, a.concentration, a.nu0, a.nu1, a.pivot, a.epsilon)

    def forward(self, log_state, sigma):
        t = sigma.reshape(-1)
        probabilities = log_state.exp().to(self.backbone.vocab_embed.embedding.dtype)
        if self.config.algo.input_mode == "argmax":
            embedded = self.backbone.get_embedding(log_state.argmax(-1))
            embedded = embedded.masked_fill(
                ~torch.isfinite(log_state).any(-1)[..., None], 0
            )
        else:
            embedded = probabilities @ self.backbone.vocab_embed.embedding
        logits = self.backbone.forward_with_embeddings(embedded, t)
        return logits.float().log_softmax(-1)

    def prior_sample(self, *batch_dims):
        t = torch.ones(batch_dims[0], device=self.device, dtype=torch.float64)
        beta = self.concentration(t)[:, None, None] * self.reference.double()
        return log_dirichlet_sample(beta.expand(*batch_dims, self.vocab_size))

    def nll(
        self,
        x0,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        clean = F.one_hot(x0, self.vocab_size).double()
        beta = corruption_parameters(
            clean, t.double(), self.concentration(t.double()), self.reference.double()
        )
        state = log_dirichlet_sample(beta)
        if valid_tokens is not None:
            state = torch.where(valid_tokens.bool()[..., None], state, clean.log())
            dropout = self.config.algo.get("condition_dropout", 0.0)
            if train_mode and dropout:
                drop = torch.rand(len(x0), device=x0.device) < dropout
                state = state.masked_fill(
                    drop[:, None, None] & ~valid_tokens.bool()[..., None], -torch.inf
                )
        log_probs = self.forward(state, t)
        loss = -log_probs.gather(-1, x0[..., None]).squeeze(-1)
        if train_mode:
            self.paper_time_sampler.observe(t, loss, valid_tokens, self.global_step)
        return loss
