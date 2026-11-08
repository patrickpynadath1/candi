"""One DiT assembled from configurable slots (see docs/components/models.md).

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
    """State @ normalized learned embedding -> preconditioning -> projection."""

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


class ViewsInput(nn.Module):
    """Several corrupted views of one sequence, embedded and fused (hybrid).

    The first view with a token table provides
    the network's token table; token feedback reads that table when its size
    matches the vocabulary, else the representation's clean embeddings.
    """

    state_kind = "views"

    def __init__(self, views, vocab_size, hidden_size, fusion=None):
        super().__init__()
        from collections.abc import Mapping

        from core.representations.noisy_inputs import CleanEmbeddingTable

        if isinstance(views, Mapping):
            views = list(views.values())
        if not views:
            raise ValueError("a views input requires at least one view")
        if any(view.representation is None for view in views):
            raise ValueError("every view requires a representation")
        self.views = nn.ModuleList(views)
        self.fusion = fusion
        token_view = next(
            (v for v in views if getattr(v.representation, "embedding", None) is not None),
            None,
        )
        # Index, not a reference: storing the Parameter here would register it twice.
        self._token_view = None if token_view is None else views.index(token_view)
        for view in views:
            bind = getattr(view.representation, "bind_embedding", None)
            if callable(bind):
                bind(self.token_table)
        if self.token_table is None or len(self.token_table) != vocab_size:
            source = views[0] if token_view is None else token_view
            self.feedback_table = CleanEmbeddingTable(source.representation, vocab_size)
        else:
            self.feedback_table = None

    @property
    def token_table(self):
        if self._token_view is None:
            return None
        return self.views[self._token_view].representation.embedding

    @property
    def table(self):
        return self.token_table if self.feedback_table is None else None

    def feedback(self):
        return self.token_table if self.feedback_table is None else self.feedback_table()

    def forward(self, states, points):
        if len(states) != len(self.views) or len(points) != len(self.views):
            raise ValueError("one state and local point are required per view")
        embeddings = [
            view.representation.to_model_embedding(state, point)
            for view, state, point in zip(self.views, states, points)
        ]
        if self.fusion is not None:
            return self.fusion(states, points, embeddings, self.views)
        return torch.stack(embeddings).sum(0)

    def clean_embedding(self, probabilities):
        return torch.stack(
            [view.representation.clean_embedding(probabilities) for view in self.views]
        ).sum(0)


# Time -----------------------------------------------------------------------


def _keep(point):
    keep = point.coefficients
    return keep[0] if isinstance(keep, tuple) else keep


TIME_COORDINATES = {
    "time": lambda time, point, epsilon: time.float(),
    "zero": lambda time, point, epsilon: torch.zeros_like(time.float()),
    # -log(keep) for keep-rate schedules (keep clamped at 1e-6).
    "negative_log_keep": lambda time, point, epsilon: -torch.log(
        _keep(point).float().clamp_min(1e-6)
    ),
    # Standalone MDLM's -log(keep): keep clamped at the dtype's smallest normal.
    "negative_log_keep_exact": lambda time, point, epsilon: -_keep(point).clamp_min(
        torch.finfo(_keep(point).dtype).tiny
    ).log(),
    # Standalone spherical flow matching's -log(keep): keep clamped at 1e-12.
    "negative_log_keep_floor": lambda time, point, epsilon: -_keep(point).clamp_min(1e-12).log(),
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

    def __init__(
        self, cond_dim, coordinate="time", adaLN=True, epsilon=1e-6, input_projection=False, time_source="global"
    ):
        super().__init__()
        if coordinate not in TIME_COORDINATES:
            raise ValueError(f"time coordinate must be one of {sorted(TIME_COORDINATES)}")
        if time_source not in {"global", "point"}:
            raise ValueError("time_source must be global or point")
        if not 0 < epsilon < 1:
            raise ValueError("time epsilon must be between zero and one")
        self.coordinate, self.adaLN, self.epsilon = coordinate, bool(adaLN), epsilon
        # "point": the first view's local time (in training, after the
        # schedule's training_time map), not the global time.
        self.time_source = time_source
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
        if self.time_source == "point":
            time = point.time
        coordinate = TIME_COORDINATES[self.coordinate](time, point, self.epsilon)
        return F.silu(self.embedder(coordinate))

    def clean_condition(self, batch, device):
        """Condition for an already-clean stream: the embedder at raw zero."""
        if not self.adaLN:
            return None
        return F.silu(self.embedder(torch.zeros(batch, device=device)))

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
        if scale == "unit_embedding":  # sqrt(state_dim), the unit-embedding scale
            scale = state_dim**0.5
        self.space, self.scale = space, float(scale)
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
    """Projection, then a dot product with a learned output embedding plus bias."""

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
        time,
        output,
        input=None,
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
        output_vocab_size=None,
        views=None,
        fusion=None,
    ):
        super().__init__(autocast_cache=autocast_cache)
        self.output_vocab_size = output_vocab_size
        if (views is None) == (input is None):
            raise ValueError("give exactly one of input (one state) or views (hybrid)")
        if views is not None:
            from functools import partial

            input = partial(ViewsInput, views, fusion=fusion)
        elif fusion is not None:
            raise ValueError("fusion requires views")
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
        # A slot may be listed with enabled=false so configs can toggle it.
        enabled = getattr(self_conditioning, "keywords", {}).get("enabled", True)
        if self_conditioning is not None and enabled:
            from functools import partial

            keywords = dict(self_conditioning.keywords)
            keywords.pop("enabled", None)
            self_conditioning = partial(self_conditioning.func, *self_conditioning.args, **keywords)
        self.self_conditioning = (
            _build(self_conditioning, **dims) if self_conditioning is not None and enabled else None
        )
        if block_execution is not None:
            self.block_execution = block_execution

    @property
    def adaLN(self):
        return self.time.adaLN

    @property
    def views(self):
        return getattr(self.input, "views", None)

    @property
    def token_table(self):
        """The token embedding table (token input or the token view's), if any."""
        return getattr(self.input, "token_table", None) if self.views is not None else getattr(self.input, "table", None)

    @property
    def output_layer(self):
        """The final AdaLN layer (the DiT name), when the head has one."""
        layer = getattr(self.output, "layer", None)
        if layer is None:
            raise AttributeError("this output head has no final AdaLN layer")
        return layer

    @property
    def conditioning_adapter(self):
        """The self-conditioning slot, under the name the coordinator checks."""
        return self.self_conditioning

    @property
    def output_width(self):
        """Logit width after any output-vocabulary slicing."""
        layer = getattr(self.output, "layer", None)
        width = layer.linear.out_features if layer is not None else len(self.output.embedding)
        return self.output_vocab_size or width

    def bind_views(self, views, *, fusion=None, fp32_clean_embedding=False):
        """Replace the token input with ``views`` that share its embedding table.

        A one-view (or hybrid) token recipe built from a token network: the first
        view's TokenRepresentation takes the input's table, so parameters, their
        order and initialization are unchanged and no random numbers are drawn.
        """
        from core.representations.noisy_inputs import TokenRepresentation

        if self.views is not None:
            raise ValueError("this network already has views")
        if type(self.input) is not TokenInput:
            raise TypeError("only a token-input network binds views; pass views to HybridDiffusion")
        if not isinstance(views[0].representation, TokenRepresentation):
            raise TypeError("the network's token table needs a TokenRepresentation view")
        table = self.input.table
        with torch.random.fork_rng(devices=[]):
            views_input = ViewsInput(views, self.vocab_size, table.shape[1], fusion=fusion)
        views[0].representation.embedding = table
        views[0].representation.fp32_clean_embedding = bool(fp32_clean_embedding)
        self.input = views_input
        return self

    def prepare(self, states, points, global_time, *, self_condition=None, apply_conditioning=True):
        """Noisy inputs -> (hidden, AdaLN condition), outside autocast."""
        with torch.autocast(states[0].device.type, enabled=False):
            if self.views is not None:
                hidden = self.input(states, points)
            elif len(states) != 1 or len(points) != 1:
                raise ValueError("this input slot drives exactly one view")
            else:
                hidden = self.input(states[0], points[0])
            if self.self_conditioning is not None and apply_conditioning:
                hidden = self.self_conditioning(hidden, self_condition)
            elif self_condition is not None:
                raise ValueError("model does not enable self-conditioning")
            condition = self.time(global_time, points[0])
            hidden = self.time.add_to_input(hidden, condition)
        return hidden, condition

    def clean_embedding(self, probabilities):
        return self.input.clean_embedding(probabilities)

    def clean_hidden(self, probabilities):
        """Inputs for an already-clean stream (block diffusion's prefix)."""
        hidden = self.clean_embedding(probabilities)
        return hidden, self.time.clean_condition(len(hidden), hidden.device)

    def clean_token_hidden(self, tokens, encode=None):
        """Clean-stream inputs from token ids for a single-state input slot.

        Token tables embed the ids; other
        slots run the encoded clean state at time 0, sigma 0, alpha 1 without
        self-conditioning.
        """
        if self.views is not None:
            width = self.output_width
            return self.clean_hidden(F.one_hot(tokens, width).float())
        if isinstance(self.input, TokenInput):
            with torch.autocast(tokens.device.type, enabled=False):
                hidden = self.input.embedding(tokens)
            return hidden, self.time.clean_condition(len(tokens), tokens.device)
        from core.modifiers.hybrid.views import ViewPoint
        from core.noise_schedules.gaussian_path import GaussianPath

        state = encode(tokens) if encode is not None else tokens
        zeros = torch.zeros(len(state), device=state.device)
        ones = torch.ones_like(zeros)
        point = ViewPoint(zeros, GaussianPath(ones, zeros, zeros, zeros))
        return self.prepare((state,), (point,), zeros, apply_conditioning=False)

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
        logits = self.output(hidden, condition)
        return logits if self.output_vocab_size is None else logits[..., : self.output_vocab_size]

    def feedback_embedding_table(self):
        if hasattr(self, "feedback_table_adapter"):
            return self.feedback_table_adapter()
        if self.views is not None:
            return self.input.feedback()
        return getattr(self.input, "table", None)
