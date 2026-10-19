"""Representations own clean encodings and noisy-state model input mappings."""

import torch
from torch import nn
from torch.nn import functional as F


class TokenRepresentation(nn.Module):
    def encode(self, tokens):
        return tokens

    def clean_state(self, tokens):
        return tokens

    def __init__(self, vocab_size, hidden_size, scale=1.0, normalize=False):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.embedding, std=hidden_size**-0.5)
        self.scale, self.normalize = float(scale), bool(normalize)

    def _table(self):
        return F.normalize(self.embedding, dim=-1) if self.normalize else self.embedding

    def to_model_embedding(self, state, point):
        del point
        return F.embedding(state, self._table()) * self.scale

    def clean_embedding(self, probabilities):
        table = self._table()
        if probabilities.shape[-1] > len(table):
            raise ValueError("posterior vocabulary exceeds representation vocabulary")
        return (
            probabilities.to(self.embedding)
            @ table[: probabilities.shape[-1]]
            * self.scale
        )


class ProjectedRepresentation(nn.Module):
    @property
    def spec(self):
        return self.state_representation.spec

    def codebook(self):
        return self.state_representation.codebook()

    def encode(self, tokens):
        return self.state_representation.encode(tokens)

    def clean_state(self, tokens):
        return self.encode(tokens)

    def posterior_mean(self, probabilities):
        return self.state_representation.posterior_mean(probabilities)

    def __init__(
        self,
        state_dim,
        hidden_size,
        scale=1.0,
        precondition=True,
        bias=True,
        state_representation=None,
        embedding_projection=False,
    ):
        super().__init__()
        if (
            state_representation is None
            or state_representation.spec.dimension != state_dim
        ):
            raise ValueError("state_dim must match the supplied state representation")
        self.state_representation = state_representation
        self.input_embedding = (
            nn.Parameter(torch.randn(state_dim, hidden_size) / hidden_size**0.5)
            if embedding_projection
            else None
        )
        self.projection = nn.Linear(
            hidden_size if embedding_projection else state_dim, hidden_size, bias=bias
        )
        self.scale, self.precondition = float(scale), bool(precondition)

    def to_model_embedding(self, state, point):
        value = state.float()
        coefficients = point.coefficients
        if self.precondition and hasattr(coefficients, "sigma"):
            scale = (coefficients.alpha.square() + coefficients.sigma.square()).rsqrt()
            value = value * scale[:, None, None]
        return self._project(value) * self.scale

    def _project(self, value):
        if self.input_embedding is not None:
            value = value @ F.normalize(self.input_embedding, dim=-1)
        return self.projection(value.to(self.projection.weight))

    def clean_embedding(self, probabilities):
        return self._project(self.posterior_mean(probabilities)) * self.scale


class CANDIRepresentation(nn.Module):
    """One-hot projection tied to CANDI's normalized token table."""

    @property
    def spec(self):
        from .spec import RepresentationSpec

        return RepresentationSpec("one_hot", self.state_dim, self.state_dim, True)

    def encode(self, tokens):
        return F.one_hot(tokens, self.state_dim).float()

    def clean_state(self, tokens):
        return self.encode(tokens.clamp_max(self.state_dim - 1))

    def posterior_mean(self, probabilities):
        return probabilities

    def __init__(self, state_dim, scale=0.5):
        super().__init__()
        self.state_dim, self.scale = int(state_dim), float(scale)
        self.embedding = None

    def bind_embedding(self, embedding):
        self.embedding = embedding

    def _table(self):
        if self.embedding is None:
            raise RuntimeError("CANDI representation is not bound to a token table")
        return self.embedding[: self.state_dim]

    def to_model_embedding(self, state, point):
        scale = (1 + point.coefficients.sigma.square()).rsqrt()
        return (state.float() @ self._table()) * scale[:, None, None] * self.scale

    def clean_embedding(self, probabilities):
        table = F.normalize(self._table(), p=2, dim=-1)
        return probabilities.to(self.embedding) @ table * self.scale


class SimplexRepresentation(TokenRepresentation):
    def clean_state(self, tokens):
        return F.one_hot(tokens, len(self.embedding)).float().log()

    def to_model_embedding(self, state, point):
        del point
        probabilities = state.exp().to(self.embedding)
        return probabilities @ self._table() * self.scale


class CleanEmbeddingTable(nn.Module):
    """Expose a representation's clean-token inputs to shared token feedback."""

    def __init__(self, representation, vocab_size):
        super().__init__()
        self.representation = representation
        self.register_buffer("probabilities", torch.eye(vocab_size), persistent=False)

    def forward(self):
        return self.representation.clean_embedding(self.probabilities)
