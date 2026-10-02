"""Continuous DiT with a learned state projection and Gaussian preconditioning."""

import torch
from torch import nn
from .prediction import make_conditioning
from .text_dit import TextDiT
from .components import LinearStateInput, EmbeddedStateInput, EmbeddingOutput
from .layers import DDiTBlock, DDiTFinalLayer, Rotary, TimestepEmbedder
from .gaussian_conditioning import (
    INPUT_PRECONDITIONING,
    TIME_CONDITIONING,
    resolve_choice,
)


class ContinuousDiT(TextDiT):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        cond_dim=128,
        n_blocks=8,
        n_heads=8,
        dropout=0.1,
        input_dim=None,
        output_dim=None,
        epsilon=1e-6,
        state_dim=None,
        input_preconditioning="gaussian",
        time_conditioning="log_one_minus_time",
        self_conditioning=False,
        self_conditioning_source="posterior_mean",
        conditioning_initialization="zero",
        input_architecture="embedding_projection",
        output_architecture="embedding_projection",
        time_input=False,
        conditioning_scale=1.0,
        input_projection_bias=False,
        input_adapter=None,
        time_adapter=None,
        output_adapter=None,
        block_execution=None,
    ):
        super().__init__(autocast_cache=False)
        if hidden_size % n_heads or (hidden_size // n_heads) % 2:
            raise ValueError("hidden_size must give even integral head dimensions")
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be between zero and one")
        if input_preconditioning == "unit_embedding" and input_architecture != "linear":
            raise ValueError(
                "unit_embedding preconditioning requires linear state input"
            )
        self.input_preconditioning = input_preconditioning
        self.time_conditioning = time_conditioning
        resolve_choice(
            INPUT_PRECONDITIONING, input_preconditioning, "input_preconditioning"
        )
        resolve_choice(TIME_CONDITIONING, time_conditioning, "time_conditioning")
        self.epsilon = epsilon
        self.vocab_size = vocab_size
        self.state_dim = vocab_size if state_dim is None else state_dim
        self.conditioning_scale = (
            self.state_dim**0.5
            if conditioning_scale == "unit_embedding"
            else float(conditioning_scale)
        )
        input_dim, output_dim = input_dim or hidden_size, output_dim or hidden_size
        if input_architecture not in {"embedding_projection", "linear"}:
            raise ValueError("unknown input architecture")
        if output_architecture not in {"embedding_projection", "adaln"}:
            raise ValueError("unknown output architecture")
        self.input_adapter = (LinearStateInput() if input_architecture == "linear"
                              else EmbeddedStateInput())
        self.input_architecture = input_architecture
        self.output_architecture = output_architecture
        if input_architecture == "linear":
            self.input_projection = nn.Linear(
                self.state_dim, hidden_size, bias=bool(input_projection_bias)
            )
        else:
            self.input_embedding = nn.Parameter(torch.empty(self.state_dim, input_dim))
            nn.init.normal_(self.input_embedding, std=input_dim**-0.5)
            with torch.no_grad():
                for _ in range(64):
                    self.input_embedding.sub_(
                        self.input_embedding.mean(0, keepdim=True)
                    )
                    self.input_embedding.div_(
                        self.input_embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    )
                self.input_embedding.sub_(self.input_embedding.mean(0, keepdim=True))
            self.input_projection = nn.Linear(input_dim, hidden_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(hidden_size, n_heads, True, cond_dim, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        if output_architecture == "adaln":
            self.output_layer = DDiTFinalLayer(hidden_size, vocab_size, cond_dim, True)
        else:
            self.output_adapter = EmbeddingOutput()
            self.output_projection = nn.Linear(hidden_size, output_dim, bias=False)
            self.output_embedding = nn.Parameter(torch.empty(vocab_size, output_dim))
            nn.init.normal_(self.output_embedding, std=output_dim**-0.5)
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.time_input_projection = (
            nn.Linear(cond_dim, hidden_size, bias=False) if time_input else None
        )

        self.conditioning_adapter = (
            make_conditioning(
                self_conditioning_source,
                self.state_dim,
                hidden_size,
                conditioning_initialization,
            )
            if self_conditioning
            else None
        )

        self.configure_components(
            input_adapter=input_adapter, time_adapter=time_adapter,
            output_adapter=output_adapter, block_execution=block_execution,
        )

    def forward(
        self,
        state,
        time,
        sigma,
        alpha,
        *,
        self_condition=None,
        return_features=False,
        attention_mask=None,
        position_ids=None,
    ):
        hidden, condition = self.prepare_hidden(
            state, time, sigma, alpha, self_condition=self_condition
        )
        return self.forward_hidden(
            hidden,
            condition,
            return_features=return_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def prepare_hidden(
        self, state, time, sigma, alpha, *, self_condition=None, apply_conditioning=True
    ):
        with torch.autocast(state.device.type, enabled=False):
            hidden = self.input_adapter(
                state, self.input_projection,
                embedding=getattr(self, "input_embedding", None),
                precondition=lambda value: INPUT_PRECONDITIONING[
                    self.input_preconditioning
                ](value, alpha.float(), sigma.float()),
            )
            if self.conditioning_adapter is not None and apply_conditioning:
                hidden = self.conditioning_adapter(
                    hidden,
                    None
                    if self_condition is None
                    else self_condition * self.conditioning_scale,
                )
            elif self_condition is not None:
                raise ValueError("model does not enable self-conditioning")
            coordinate = TIME_CONDITIONING[self.time_conditioning](
                time.float(), sigma.float(), alpha.float(), self.epsilon
            )
            condition = self.time_adapter(coordinate, self.sigma_map)
            if self.time_input_projection is not None:
                hidden = hidden + self.time_input_projection(condition)[:, None, :]
        return hidden, condition

    def clean_state_hidden(self, state):
        zeros = torch.zeros(len(state), device=state.device)
        ones = torch.ones_like(zeros)
        return self.prepare_hidden(state, zeros, zeros, ones, apply_conditioning=False)

    def project_hidden(self, hidden, condition):
        if self.output_architecture == "adaln":
            return super().project_hidden(hidden, condition)
        return self.output_adapter(
            hidden, condition, projection=self.output_projection,
            embedding=self.output_embedding, bias=self.output_bias,
        )
