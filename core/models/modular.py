"""One DiT assembled from configurable slots (see docs/design/modular-dit.md).

Slots own their learned weights. The network builds each slot with the
dimensions it needs, in the legacy construction order (input, time embedder,
blocks, output, time input projection, self-conditioning), so the legacy model
classes are presets of this one with identical initialization.

All slots run outside autocast, as the legacy input preparation did.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .components import LayerOutput, TemperatureOutput
from .gaussian_conditioning import INPUT_PRECONDITIONING, TIME_CONDITIONING, resolve_choice
from .layers import DDiTBlock, DDiTFinalLayer, EmbeddingLayer, Rotary, TimestepEmbedder
from .prediction import make_conditioning
from .text_dit import TextDiT


def _path(point):
    """(alpha, sigma) of a Gaussian-style point, as float32 vectors."""
    coefficients = point.coefficients
    return coefficients.alpha.float(), coefficients.sigma.float()


# Inputs ---------------------------------------------------------------------


class TokenInput(nn.Module):
    """Token ids (or a [B, L, V] logit state, via softmax) -> embeddings."""

    state_kind = "tokens"

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = EmbeddingLayer(hidden_size, vocab_size)

    @property
    def table(self):
        return self.embedding.embedding

    def forward(self, state, point):
        return self.embedding(state)


class SimplexInput(TokenInput):
    """Simplex log-states: expected embedding, or the argmax token's."""

    state_kind = "simplex"

    def __init__(self, vocab_size, hidden_size, mode="expectation", precision="legacy", chunk_size=0):
        super().__init__(vocab_size, hidden_size)
        if mode not in {"expectation", "argmax"}:
            raise ValueError("simplex input mode must be expectation or argmax")
        self.mode, self.precision, self.chunk_size = mode, precision, chunk_size

    def forward(self, state, point):
        if self.mode == "argmax":
            return self.embedding(state.argmax(-1)).masked_fill(
                ~torch.isfinite(state).any(-1)[..., None], 0
            )
        from .simplex_projection import simplex_projection

        return simplex_projection(state, self.table, self.precision, self.chunk_size)


class LinearInput(nn.Module):
    """Continuous state -> optional path preconditioning -> linear projection."""

    state_kind = "continuous"

    def __init__(self, state_dim, hidden_size, bias=True, preconditioning="none"):
        super().__init__()
        resolve_choice(INPUT_PRECONDITIONING, preconditioning, "preconditioning")
        self.state_dim, self.preconditioning = state_dim, preconditioning
        self.projection = nn.Linear(state_dim, hidden_size, bias=bool(bias))

    def precondition(self, hidden, point):
        if self.preconditioning == "none":
            return hidden
        alpha, sigma = _path(point)
        return INPUT_PRECONDITIONING[self.preconditioning](hidden, alpha, sigma)

    def forward(self, state, point):
        return self.projection(self.precondition(state.float(), point))


class EmbeddedProjectionInput(LinearInput):
    """ContinuousDiT's default: state @ normalized embedding -> projection."""

    def __init__(self, state_dim, hidden_size, input_dim=None, preconditioning="gaussian"):
        nn.Module.__init__(self)
        resolve_choice(INPUT_PRECONDITIONING, preconditioning, "preconditioning")
        input_dim = input_dim or hidden_size
        self.state_dim, self.preconditioning = state_dim, preconditioning
        self.embedding = nn.Parameter(torch.empty(state_dim, input_dim))
        nn.init.normal_(self.embedding, std=input_dim**-0.5)
        with torch.no_grad():
            for _ in range(64):
                self.embedding.sub_(self.embedding.mean(0, keepdim=True))
                self.embedding.div_(self.embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            self.embedding.sub_(self.embedding.mean(0, keepdim=True))
        self.projection = nn.Linear(input_dim, hidden_size)

    def forward(self, state, point):
        hidden = state.float() @ F.normalize(self.embedding.float(), dim=-1)
        return self.projection(self.precondition(hidden, point))


# Time -----------------------------------------------------------------------


def _keep(point):
    keep = point.coefficients
    return keep[0] if isinstance(keep, tuple) else keep


TIME_COORDINATES = {
    "time": lambda time, point, epsilon: time.float(),
    "zero": lambda time, point, epsilon: torch.zeros_like(time.float()),
    # -log(keep) for keep-rate schedules (MDLM / uniform with time conditioning).
    "negative_log_keep": lambda time, point, epsilon: -_keep(point).clamp_min(
        torch.finfo(_keep(point).dtype).tiny
    ).log(),
    # HybridModel's log(1 - t).
    "log1p_negative_time": lambda time, point, epsilon: torch.log1p(
        -time.float().clamp(max=1 - 1e-6)
    ),
    **{
        name: (
            lambda function: lambda time, point, epsilon: function(
                time.float(), _path(point)[1], _path(point)[0], epsilon
            )
        )(function)
        for name, function in TIME_CONDITIONING.items()
        if name != "time"
    },
}


class TimeConditioning(nn.Module):
    """Global time (or a path coefficient) -> AdaLN conditioning vector."""

    def __init__(self, cond_dim, coordinate="time", adaLN=True, epsilon=1e-6, input_projection=False):
        super().__init__()
        if coordinate not in TIME_COORDINATES:
            raise ValueError(f"time coordinate must be one of {sorted(TIME_COORDINATES)}")
        self.coordinate, self.adaLN, self.epsilon = coordinate, bool(adaLN), epsilon
        self.wants_input_projection = bool(input_projection)
        # Kept even without AdaLN, for identical initialization and checkpoints.
        self.embedder = TimestepEmbedder(cond_dim)
        self.input_projection = None

    def build_input_projection(self, cond_dim, hidden_size):
        if self.wants_input_projection:
            self.input_projection = nn.Linear(cond_dim, hidden_size, bias=False)

    def forward(self, time, point):
        if not self.adaLN:
            return None
        coordinate = TIME_COORDINATES[self.coordinate](time, point, self.epsilon)
        return F.silu(self.embedder(coordinate))

    def add_to_input(self, hidden, condition):
        if self.input_projection is None:
            return hidden
        return hidden + self.input_projection(condition)[:, None, :]


# Self-conditioning ----------------------------------------------------------


class SelfConditioning(nn.Module):
    """Add a projection of the previous prediction to the input hidden vectors."""

    def __init__(self, hidden_size, state_dim=None, source="posterior_mean", space="hidden", initialization="zero", scale=1.0):
        super().__init__()
        if space not in {"hidden", "state"}:
            raise ValueError("self-conditioning space must be hidden or state")
        if space == "state" and state_dim is None:
            raise ValueError("state-space self-conditioning needs the state dimension")
        self.space, self.scale = space, scale
        self.adapter = make_conditioning(
            source, state_dim if space == "state" else hidden_size, hidden_size, initialization
        )

    def forward(self, hidden, self_condition):
        if self_condition is not None and self.scale != 1.0:
            self_condition = self_condition * self.scale
        return self.adapter(hidden, self_condition)


# Outputs --------------------------------------------------------------------


class AdaLNHead(nn.Module):
    """Final AdaLN layer; optionally a learned per-token temperature column."""

    def __init__(self, hidden_size, vocab_size, cond_dim, adaLN=True, temperature=False):
        super().__init__()
        self.temperature = bool(temperature)
        self.layer = DDiTFinalLayer(hidden_size, vocab_size + int(self.temperature), cond_dim, adaLN)
        self.apply_output = TemperatureOutput() if self.temperature else LayerOutput()

    def tie(self, table):
        self.layer.linear.weight = table

    def forward(self, hidden, condition):
        return self.apply_output(hidden, condition, layer=self.layer)


class EmbeddingHead(nn.Module):
    """ContinuousDiT's head: projection, then dot with an output embedding."""

    def __init__(self, hidden_size, vocab_size, output_dim=None):
        super().__init__()
        output_dim = output_dim or hidden_size
        self.projection = nn.Linear(hidden_size, output_dim, bias=False)
        self.embedding = nn.Parameter(torch.empty(vocab_size, output_dim))
        nn.init.normal_(self.embedding, std=output_dim**-0.5)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden, condition):
        return self.projection(hidden) @ self.embedding.t() + self.bias


# Network --------------------------------------------------------------------


def _build(factory, **dims):
    """Call a Hydra partial (or class) with the dimensions it accepts."""
    import inspect

    target = getattr(factory, "func", factory)
    accepted = inspect.signature(target).parameters
    given = getattr(factory, "keywords", {})
    return factory(**{k: v for k, v in dims.items() if k in accepted and k not in given})


class ModularDiT(TextDiT):
    """A DiT whose input, time, self-conditioning and output are slots."""

    def __init__(
        self,
        vocab_size,
        input,
        time,
        output,
        hidden_size=512,
        cond_dim=128,
        n_blocks=8,
        n_heads=8,
        dropout=0.1,
        state_dim=None,
        self_conditioning=None,
        block_adaLN=True,
        tie_word_embeddings=False,
        sublayer_factory=None,
        mlp_hidden_size=None,
        block_execution=None,
        autocast_cache=True,
    ):
        super().__init__(autocast_cache=autocast_cache)
        if hidden_size % n_heads or (hidden_size // n_heads) % 2:
            raise ValueError("hidden_size must give an even integral head dimension")
        dims = dict(vocab_size=vocab_size, hidden_size=hidden_size, cond_dim=cond_dim, state_dim=state_dim)
        self.vocab_size, self.state_dim = vocab_size, state_dim
        self.input = _build(input, **dims)
        self.time = _build(time, **dims)
        self.rotary_emb = Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    hidden_size,
                    n_heads,
                    block_adaLN,
                    cond_dim,
                    dropout=dropout,
                    sublayer_factory=sublayer_factory,
                    mlp_hidden_size=mlp_hidden_size,
                )
                for _ in range(n_blocks)
            ]
        )
        self.output = _build(output, **dims)
        if tie_word_embeddings:
            self.output.tie(self.input.table)
        self.time.build_input_projection(cond_dim, hidden_size)
        self.self_conditioning = (
            None if self_conditioning is None else _build(self_conditioning, **dims)
        )
        if block_execution is not None:
            self.block_execution = block_execution

    @property
    def adaLN(self):
        return self.time.adaLN

    def prepare(self, states, points, global_time, *, self_condition=None):
        if len(states) != 1 or len(points) != 1:
            raise ValueError("this input slot drives exactly one view")
        state, point = states[0], points[0]
        with torch.autocast(state.device.type, enabled=False):
            hidden = self.input(state, point)
            if self.self_conditioning is not None:
                hidden = self.self_conditioning(hidden, self_condition)
            elif self_condition is not None:
                raise ValueError("model does not enable self-conditioning")
            condition = self.time(global_time, point)
            hidden = self.time.add_to_input(hidden, condition)
        return hidden, condition

    def forward(
        self,
        states,
        points,
        global_time,
        *,
        self_condition=None,
        return_features=False,
        attention_mask=None,
        position_ids=None,
        hidden_only=False,
    ):
        hidden, condition = self.prepare(states, points, global_time, self_condition=self_condition)
        return self.forward_hidden(
            hidden,
            condition,
            return_features=return_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hidden_only=hidden_only,
        )

    def project_hidden(self, hidden, condition):
        return self.output(hidden, condition)

    def feedback_embedding_table(self):
        if hasattr(self, "feedback_table_adapter"):
            return self.feedback_table_adapter()
        return getattr(self.input, "table", None)
