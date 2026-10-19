"""Shared transformer execution, independent of the noisy-state representation."""

from contextlib import nullcontext

import torch
from torch import nn

from .components import (
    BlockExecutionOutput,
    DiffusionBlocks,
    LayerOutput,
    TimeConditioning,
)
from .prediction import ModelPrediction


class TextDiT(nn.Module):
    """Backbone used by the public method-specific construction recipes.

    Constructors install blocks, positional encoding, learned input/output
    modules, and parameter-free execution components at their historical paths.
    This preserves serialized weights while making execution interchangeable.
    Input preparation remains outside the backbone: callers can supply token,
    dense, simplex, or already-fused hybrid hidden vectors.
    """

    def __init__(self, *, autocast_cache=True):
        super().__init__()
        self.block_execution = DiffusionBlocks()
        self.output_adapter = LayerOutput()
        self.time_adapter = TimeConditioning()
        self.autocast_cache = autocast_cache

    def configure_components(
        self,
        *,
        input_adapter=None,
        time_adapter=None,
        output_adapter=None,
        block_execution=None,
    ):
        """Replace execution components without relocating learned tensors.

        Hydra recursively instantiates these optional constructor arguments.
        Each replacement follows its corresponding component call signature.
        """
        for name, component in (
            ("input_adapter", input_adapter),
            ("time_adapter", time_adapter),
            ("output_adapter", output_adapter),
            ("block_execution", block_execution),
        ):
            if component is not None:
                if not isinstance(component, nn.Module):
                    raise TypeError(f"{name} must be a torch.nn.Module")
                setattr(self, name, component)

    def forward_hidden(
        self,
        hidden,
        condition,
        *,
        return_features=False,
        attention_mask=None,
        position_ids=None,
        hidden_only=False,
    ):
        # RoPE frequencies must be constructed in FP32 even under autocast.
        with torch.autocast(hidden.device.type, enabled=False):
            rotary = self.rotary_emb(hidden, position_ids=position_ids)
        # Continuous bootstrap must not cache detached casts for a later pass.
        precision = (
            nullcontext()
            if self.autocast_cache
            else torch.autocast(
                hidden.device.type,
                enabled=torch.is_autocast_enabled(hidden.device.type),
                cache_enabled=False,
            )
        )
        with precision:
            hidden = self.block_execution(
                hidden,
                self.blocks,
                rotary,
                condition=condition,
                attention_mask=attention_mask,
                project=self.project_hidden,
                feedback_embedding=self.feedback_embedding_table(),
            )
            auxiliary_logits = None
            if isinstance(hidden, BlockExecutionOutput):
                auxiliary_logits = hidden.auxiliary_logits
                hidden = hidden.hidden
            if hidden_only and not return_features:
                raise ValueError("hidden_only requires return_features")
            logits = None if hidden_only else self.project_hidden(hidden, condition)
        if return_features:
            return ModelPrediction(logits, hidden, auxiliary_logits=auxiliary_logits)
        return logits

    def project_hidden(self, hidden, condition):
        return self.output_adapter(hidden, condition, layer=self.output_layer)

    def feedback_embedding_table(self):
        if hasattr(self, "feedback_table_adapter"):
            return self.feedback_table_adapter()
        embedding = getattr(getattr(self, "vocab_embed", None), "embedding", None)
        return embedding
