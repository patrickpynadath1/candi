"""One shared transformer over a sum of parallel branch embeddings."""

import torch
from torch import nn
from torch.nn import functional as F

from .prediction import ModelPrediction, make_conditioning


class TokenEmbeddingAdapter(nn.Module):
    def __init__(self, vocab_size, hidden_size, scale=1.0, normalize=False):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.embedding, std=hidden_size**-0.5)
        self.scale, self.normalize = float(scale), bool(normalize)

    def _table(self):
        return F.normalize(self.embedding, dim=-1) if self.normalize else self.embedding

    def forward(self, state, point):
        del point
        return F.embedding(state, self._table()) * self.scale

    def clean_embedding(self, probabilities):
        table = self._table()
        if probabilities.shape[-1] > len(table):
            raise ValueError("posterior vocabulary exceeds adapter vocabulary")
        return probabilities.to(self.embedding) @ table[:probabilities.shape[-1]] * self.scale


class ContinuousInputAdapter(nn.Module):
    def __init__(self, state_dim, hidden_size, scale=1.0, precondition=True, bias=True):
        super().__init__()
        self.projection = nn.Linear(state_dim, hidden_size, bias=bias)
        self.scale, self.precondition = float(scale), bool(precondition)

    def forward(self, state, point):
        value = state.float()
        coefficients = point.coefficients
        if self.precondition and hasattr(coefficients, "sigma"):
            scale = (coefficients.alpha.square() + coefficients.sigma.square()).rsqrt()
            value = value * scale[:, None, None]
        return self.projection(value) * self.scale

    def clean_embedding(self, probabilities):
        if probabilities.shape[-1] != self.projection.in_features:
            raise ValueError("continuous adapter clean posterior must match state dimension")
        return self.projection(probabilities.to(self.projection.weight)) * self.scale


class CANDIContinuousAdapter(nn.Module):
    """One-hot projection tied to CANDI's normalized token table."""

    def __init__(self, state_dim, scale=0.5):
        super().__init__()
        self.state_dim, self.scale = int(state_dim), float(scale)
        self.embedding = None

    def bind_embedding(self, embedding):
        self.embedding = embedding

    def _table(self):
        if self.embedding is None:
            raise RuntimeError("CANDI continuous adapter is not bound to a token table")
        return self.embedding[:self.state_dim]

    def forward(self, state, point):
        scale = (1 + point.coefficients.sigma.square()).rsqrt()
        return (state.float() @ self._table()) * scale[:, None, None] * self.scale

    def clean_embedding(self, probabilities):
        table = F.normalize(self._table(), p=2, dim=-1)
        return probabilities.to(self.embedding) @ table * self.scale


class CANDIFusion(nn.Module):
    """Legacy CANDI fusion: clean sites bypass noisy preconditioning."""

    def __init__(self, mask_index, discrete_branch=0):
        super().__init__()
        self.mask_index, self.discrete_branch = int(mask_index), int(discrete_branch)

    def forward(self, states, points, embeddings, branches):
        del embeddings
        tokens = states[self.discrete_branch]
        adapter = branches[self.discrete_branch].adapter
        table = adapter.embedding.float()
        normalized = F.normalize(table, p=2, dim=-1)
        clean = F.embedding(
            tokens.clamp_max(self.mask_index - 1), normalized[:-1]
        )
        mask = table[self.mask_index]
        mask = mask / mask.norm().clamp_min(1e-12)
        continuous = (
            states[1].float() @ normalized[:states[1].shape[-1]].float()
        ).to(states[1].dtype)
        sigma = points[1].coefficients.sigma.float()
        c_in = 1 / (1 + sigma.square()).sqrt()
        continuous_scale = branches[1].adapter.scale
        noisy = (
            adapter.scale * mask.view(1, 1, -1)
            + continuous_scale * continuous * c_in[:, None, None]
        )
        return torch.where(tokens.ne(self.mask_index)[..., None], clean, noisy)


class SimplexInputAdapter(TokenEmbeddingAdapter):
    def forward(self, state, point):
        del point
        probabilities = state.exp().to(self.embedding)
        return probabilities @ self._table() * self.scale


class HybridModel(nn.Module):
    """Fuses branch inputs once, then performs exactly one network evaluation."""

    supports_fused_streams = True

    def __init__(self, backbone, branches, self_conditioning=False,
                 self_conditioning_source="posterior_mean",
                 conditioning_initialization="zero", fusion=None,
                 time_conditioning="identity", output_vocab_size=None):
        super().__init__()
        if not branches:
            raise ValueError("hybrid model requires at least one branch")
        for branch in branches:
            if branch.adapter is None:
                raise ValueError("every hybrid branch requires an input adapter")
        self.backbone = backbone
        self.branches = nn.ModuleList(branches)
        self.fusion = fusion
        if time_conditioning not in {"identity", "negative_log_keep"}:
            raise ValueError("unknown hybrid time conditioning")
        self.time_conditioning = time_conditioning
        self.output_vocab_size = output_vocab_size
        token_embedding = getattr(branches[0].adapter, "embedding", None)
        if token_embedding is not None and hasattr(backbone, "vocab_embed"):
            backbone.vocab_embed.embedding = token_embedding
        for branch in branches[1:]:
            bind = getattr(branch.adapter, "bind_embedding", None)
            if callable(bind):
                bind(token_embedding)
        hidden = backbone.output_layer.linear.in_features
        self.conditioning_adapter = (
            make_conditioning(
                self_conditioning_source, hidden, hidden,
                conditioning_initialization,
            ) if self_conditioning else None
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)

    def fused_embedding(self, states, points):
        if len(states) != len(self.branches) or len(points) != len(self.branches):
            raise ValueError("one state and local point are required per branch")
        with torch.autocast(states[0].device.type, enabled=False):
            embeddings = [
                branch.embed(state, point)
                for branch, state, point in zip(self.branches, states, points)
            ]
            return (
                self.fusion(states, points, embeddings, self.branches)
                if self.fusion is not None else torch.stack(embeddings).sum(0)
            )

    def clean_embedding(self, probabilities):
        return torch.stack([
            branch.clean_embedding(probabilities) for branch in self.branches
        ]).sum(0)

    def prepare_fused(self, states, points, global_time, self_condition=None,
                      apply_conditioning=True):
        hidden = self.fused_embedding(states, points)
        with torch.autocast(hidden.device.type, enabled=False):
            if self.conditioning_adapter is not None and apply_conditioning:
                hidden = self.conditioning_adapter(hidden, self_condition)
            elif self_condition is not None:
                raise ValueError("model does not enable self-conditioning")
            coordinate = global_time.float()
            if self.time_conditioning == "negative_log_keep":
                keep = points[0].coefficients
                if isinstance(keep, tuple):
                    keep = keep[0]
                coordinate = -torch.log(keep.float().clamp_min(1e-6))
            condition = F.silu(self.backbone.sigma_map(coordinate)) \
                if self.backbone.adaLN else None
        return hidden, condition

    def forward(self, states, points, global_time, *, self_condition=None,
                return_features=False, attention_mask=None, position_ids=None):
        hidden, condition = self.prepare_fused(
            states, points, global_time, self_condition=self_condition
        )
        return self._slice_output(self.backbone.forward_hidden(
            hidden, condition, return_features=return_features,
            attention_mask=attention_mask, position_ids=position_ids,
        ))

    def _slice_output(self, output):
        if self.output_vocab_size is None:
            return output
        if isinstance(output, ModelPrediction):
            return ModelPrediction(
                output.logits[..., :self.output_vocab_size], output.hidden,
                None if output.clean_logits is None else output.clean_logits[..., :self.output_vocab_size],
                output.clean_hidden,
            )
        return output[..., :self.output_vocab_size]

    def forward_hidden(self, *args, **kwargs):
        return self._slice_output(self.backbone.forward_hidden(*args, **kwargs))

    def project_hidden(self, *args, **kwargs):
        return self._slice_output(self.backbone.project_hidden(*args, **kwargs))

    def legacy_candi_state_map(self):
        """Strict model-key mapping used by the legacy checkpoint importer."""
        mapping = {}
        for key in self.state_dict():
            if key.startswith("backbone."):
                mapping[key] = key
            elif key == "branches.0.adapter.embedding":
                mapping[key] = "backbone.vocab_embed.embedding"
            elif key == "branches.1.adapter.embedding":
                mapping[key] = "backbone.vocab_embed.embedding"
        return mapping

    def clean_state_hidden(self, probabilities):
        hidden = self.clean_embedding(probabilities)
        zeros = torch.zeros(len(hidden), device=hidden.device)
        condition = F.silu(self.backbone.sigma_map(zeros)) if self.backbone.adaLN else None
        return hidden, condition
