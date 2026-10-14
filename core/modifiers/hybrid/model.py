"""One shared transformer over a sum of parallel view embeddings."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from core.models.prediction import ModelPrediction, make_conditioning
from core.representations.noisy_inputs import CleanEmbeddingTable


class CANDIFusion(nn.Module):
    """Legacy CANDI fusion: clean sites bypass noisy preconditioning."""

    def __init__(self, mask_index, discrete_view=0):
        super().__init__()
        self.mask_index, self.discrete_view = int(mask_index), int(discrete_view)

    def forward(self, states, points, embeddings, views):
        del embeddings
        tokens = states[self.discrete_view]
        representation = views[self.discrete_view].representation
        table = representation.embedding.float()
        normalized = F.normalize(table, p=2, dim=-1)
        clean = F.embedding(tokens.clamp_max(self.mask_index - 1), normalized[:-1])
        mask = table[self.mask_index]
        mask = mask / mask.norm().clamp_min(1e-12)
        continuous = (states[1].float() @ normalized[: states[1].shape[-1]].float()).to(
            states[1].dtype
        )
        sigma = points[1].coefficients.sigma.float()
        c_in = 1 / (1 + sigma.square()).sqrt()
        continuous_scale = views[1].representation.scale
        noisy = (
            representation.scale * mask.view(1, 1, -1)
            + continuous_scale * continuous * c_in[:, None, None]
        )
        return torch.where(tokens.ne(self.mask_index)[..., None], clean, noisy)


class HybridModel(nn.Module):
    """Fuses view inputs once, then performs exactly one network evaluation."""

    supports_fused_streams = True

    def __init__(
        self,
        backbone,
        views,
        self_conditioning=False,
        self_conditioning_source="posterior_mean",
        conditioning_initialization="zero",
        fusion=None,
        time_conditioning="identity",
        output_vocab_size=None,
    ):
        super().__init__()
        if isinstance(views, Mapping):
            views = list(views.values())
        if not views:
            raise ValueError("hybrid model requires at least one view")
        for view in views:
            if view.representation is None:
                raise ValueError("every hybrid view requires a representation")
        self.backbone = backbone
        self.views = nn.ModuleList(views)
        self.fusion = fusion
        if time_conditioning not in {
            "identity",
            "negative_log_keep",
            "log_one_minus_time",
        }:
            raise ValueError("unknown hybrid time conditioning")
        self.time_conditioning = time_conditioning
        self.output_vocab_size = output_vocab_size
        token_view = next(
            (
                view
                for view in views
                if getattr(view.representation, "embedding", None) is not None
            ),
            None,
        )
        token_embedding = (
            None if token_view is None else token_view.representation.embedding
        )
        if hasattr(backbone, "vocab_embed"):
            vocab_size = backbone.vocab_embed.embedding.shape[0]
            if token_embedding is not None:
                backbone.vocab_embed.embedding = token_embedding
                if len(token_embedding) != vocab_size:
                    backbone.feedback_table_adapter = CleanEmbeddingTable(
                        token_view.representation, vocab_size
                    )
            else:
                # Continuous views already own the trainable input mapping.
                # Avoid leaving an unused token table in the optimizer/DDP graph.
                del backbone.vocab_embed
                backbone.feedback_table_adapter = CleanEmbeddingTable(
                    views[0].representation, vocab_size
                )
        for view in views:
            bind = getattr(view.representation, "bind_embedding", None)
            if callable(bind):
                bind(token_embedding)
        hidden = backbone.blocks[0].attn_qkv.in_features
        self.conditioning_adapter = (
            make_conditioning(
                self_conditioning_source,
                hidden,
                hidden,
                conditioning_initialization,
            )
            if self_conditioning
            else None
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)

    def fused_embedding(self, states, points):
        if len(states) != len(self.views) or len(points) != len(self.views):
            raise ValueError("one state and local point are required per view")
        with torch.autocast(states[0].device.type, enabled=False):
            embeddings = [
                view.representation.to_model_embedding(state, point)
                for view, state, point in zip(self.views, states, points)
            ]
            return (
                self.fusion(states, points, embeddings, self.views)
                if self.fusion is not None
                else torch.stack(embeddings).sum(0)
            )

    def clean_embedding(self, probabilities):
        return torch.stack(
            [view.representation.clean_embedding(probabilities) for view in self.views]
        ).sum(0)

    def prepare_fused(
        self, states, points, global_time, self_condition=None, apply_conditioning=True
    ):
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
            elif self.time_conditioning == "log_one_minus_time":
                coordinate = torch.log1p(-global_time.float().clamp(max=1 - 1e-6))
            condition = (
                self.backbone.time_adapter(coordinate, self.backbone.sigma_map)
                if self.backbone.adaLN
                else None
            )
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
    ):
        hidden, condition = self.prepare_fused(
            states, points, global_time, self_condition=self_condition
        )
        return self._slice_output(
            self.backbone.forward_hidden(
                hidden,
                condition,
                return_features=return_features,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        )

    def _slice_output(self, output):
        if self.output_vocab_size is None:
            return output
        if isinstance(output, ModelPrediction):
            return ModelPrediction(
                output.logits[..., : self.output_vocab_size],
                output.hidden,
                auxiliary_logits=None
                if output.auxiliary_logits is None
                else output.auxiliary_logits[..., : self.output_vocab_size],
            )
        return output[..., : self.output_vocab_size]

    def forward_hidden(self, *args, **kwargs):
        return self._slice_output(self.backbone.forward_hidden(*args, **kwargs))

    def project_hidden(self, *args, **kwargs):
        return self._slice_output(self.backbone.project_hidden(*args, **kwargs))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Existing core CANDI/uniform-simplex checkpoints retain identical tensors.
        old_prefix = prefix + "branches."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                suffix = key[len(old_prefix) :]
                if ".adapter." in suffix:
                    suffix = suffix.replace(".adapter.", ".representation.", 1)
                else:
                    suffix = suffix.replace(
                        ".representation.", ".representation.state_representation.", 1
                    )
                state_dict[prefix + "views." + suffix] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def legacy_candi_state_map(self):
        """Strict model-key mapping used by the legacy checkpoint importer."""
        mapping = {}
        for key in self.state_dict():
            if key.startswith("backbone."):
                mapping[key] = key
            elif key == "views.0.representation.embedding":
                mapping[key] = "backbone.vocab_embed.embedding"
            elif key == "views.1.representation.embedding":
                mapping[key] = "backbone.vocab_embed.embedding"
        return mapping

    def clean_state_hidden(self, probabilities):
        hidden = self.clean_embedding(probabilities)
        zeros = torch.zeros(len(hidden), device=hidden.device)
        condition = (
            F.silu(self.backbone.sigma_map(zeros)) if self.backbone.adaLN else None
        )
        return hidden, condition
