"""Final-hidden-state encoder/decoder execution for block diffusion.

Clean tokens are encoded once. Decoder layers jointly attend to their noisy
states and the same final clean representation, as in E2D2 (2510.22852).
Visibility is supplied by the block layout, independently of corruption/loss.
"""

from contextlib import nullcontext
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from core.models.components import DiffusionBlocks
from core.models.layers import DDiTBlock
from core.models.prediction import ModelPrediction
from core.models.kv_cache import append_key_value
from .readout import EncoderKeyValueReadout


class EncoderDecoderModel(nn.Module):
    """Supply clean context to a denoising backbone.

    ``encoder_layers=0`` uses the clean input embeddings directly: there is
    no contextual encoder. Decoder layers still normalize/project their own
    context K/V and apply positional encoding and the shared visibility mask.
    Positive depths encode the clean inputs before those same operations.
    """

    def __init__(self, model, encoder_layers, context_attention=None):
        super().__init__()
        if type(encoder_layers) is not int or encoder_layers < 0:
            raise ValueError("encoder_layers must be a nonnegative integer")
        if not model.blocks or not all(
            isinstance(block, DDiTBlock) for block in model.blocks
        ):
            raise TypeError(
                "encoder-decoder execution requires DDiTBlock decoder layers"
            )
        if any(not block.positionwise_sublayer for block in model.blocks):
            raise ValueError(
                "encoder-decoder execution requires positionwise MLP sublayers"
            )
        if type(model.block_execution) is not DiffusionBlocks:
            raise ValueError(
                "encoder-decoder execution requires the standard decoder block policy"
            )
        if context_attention is not None and encoder_layers == 0:
            raise ValueError(
                "encoder K/V attention requires at least one encoder layer"
            )
        self.context_attention = context_attention
        self.model = model
        first = model.blocks[0]
        hidden_size = first.attn_qkv.in_features
        n_heads, dropout = first.n_heads, first.dropout
        self.activation_checkpointing = False
        full_layers = (
            encoder_layers if context_attention is None else encoder_layers - 1
        )
        self.encoder = nn.ModuleList(
            DDiTBlock(hidden_size, n_heads, adaLN=False, dropout=dropout)
            for _ in range(full_layers)
        )
        if context_attention is not None:
            self.encoder.append(EncoderKeyValueReadout(hidden_size, n_heads))
        self.encoder_execution = DiffusionBlocks()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _decode(self, block, hidden, rotary, condition, mask, context, context_rotary):
        if self.context_attention is None:
            prefix = (
                block.project_context_key_value(context, context_rotary)
                if context.shape[1]
                else None
            )
        else:
            prefix = context
        return block(
            hidden,
            rotary,
            condition,
            mask,
            key_value_prefix=prefix,
            context_attention=self.context_attention,
        )

    def _encode(self, clean, rotary, mask):
        if self.context_attention is None:
            return (
                self.encoder_execution(clean, self.encoder, rotary, attention_mask=mask)
                if clean.shape[1]
                else clean
            )
        if not clean.shape[1]:
            return None
        hidden, pairs = clean, []
        for block in self.encoder:
            if self.activation_checkpointing and torch.is_grad_enabled():
                scope = (
                    nullcontext()
                    if torch.compiler.is_compiling()
                    else set_checkpoint_early_stop(False)
                )
                with scope:
                    hidden, pair = checkpoint(
                        block,
                        hidden,
                        rotary,
                        None,
                        mask,
                        return_key_value=True,
                        use_reentrant=False,
                    )
            else:
                hidden, pair = block(
                    hidden, rotary, attention_mask=mask, return_key_value=True
                )
            pairs.append(pair)
        return tuple(torch.stack(parts, dim=2) for parts in zip(*pairs))

    def forward(self, clean, noisy, condition, layout, cache=None):
        if cache is not None and not self.training and not torch.is_grad_enabled():
            with torch.autocast(noisy.device.type, enabled=False):
                rotary = self.model.rotary_emb(
                    noisy, position_ids=layout.position_ids[:, clean.shape[1] :]
                )
            return self._cached_forward(clean, noisy, condition, layout, rotary, cache)
        return self.forward_uncached(clean, noisy, condition, layout)

    def forward_uncached(self, clean, noisy, condition, layout):
        """Pure tensor execution; compilation does not capture mutable caches."""
        backbone = self.model
        clean_length = clean.shape[1]
        with torch.autocast(noisy.device.type, enabled=False):
            noisy_rotary = backbone.rotary_emb(
                noisy, position_ids=layout.position_ids[:, clean_length:]
            )
            clean_rotary = (
                backbone.rotary_emb(
                    clean, position_ids=layout.position_ids[:, :clean_length]
                )
                if clean_length
                else None
            )
        context = self._encode(
            clean,
            clean_rotary,
            layout.attention_mask[:, :, :clean_length, :clean_length],
        )
        mask = layout.attention_mask[:, :, clean_length:, :]
        for block in backbone.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                scope = (
                    nullcontext()
                    if torch.compiler.is_compiling()
                    else set_checkpoint_early_stop(False)
                )
                with scope:
                    noisy = checkpoint(
                        partial(self._decode, block),
                        noisy,
                        noisy_rotary,
                        condition,
                        mask,
                        context,
                        clean_rotary,
                        use_reentrant=False,
                    )
            else:
                noisy = self._decode(
                    block, noisy, noisy_rotary, condition, mask, context, clean_rotary
                )
        return ModelPrediction(backbone.project_hidden(noisy, condition), noisy)

    def _cached_forward(self, clean, noisy, condition, layout, noisy_rotary, cache):
        """Extend encoder and decoder K/V only for newly completed clean tokens."""
        length = clean.shape[1]
        encoded = cache.get("clean_length", 0)
        encoder_kv = cache.setdefault("encoder_kv", [None] * len(self.encoder))
        decoder_kv = cache.setdefault(
            "decoder_context_kv", [None] * len(self.model.blocks)
        )
        if length > encoded:
            hidden = clean[:, encoded:]
            with torch.autocast(clean.device.type, enabled=False):
                rotary = self.model.rotary_emb(
                    hidden, position_ids=layout.position_ids[:, encoded:length]
                )
            mask = layout.attention_mask[:, :, encoded:length, :length]
            for index, block in enumerate(self.encoder):
                hidden, new_kv = block(
                    hidden,
                    rotary,
                    None,
                    mask,
                    key_value_prefix=encoder_kv[index],
                    return_key_value=True,
                )
                encoder_kv[index] = append_key_value(encoder_kv[index], new_kv)
            if self.context_attention is None:
                for index, block in enumerate(self.model.blocks):
                    new_kv = block.project_context_key_value(hidden, rotary)
                    decoder_kv[index] = append_key_value(decoder_kv[index], new_kv)
            else:
                # One shared cache of actual encoder K/V, not copies per decoder.
                cache["encoder_depth_kv"] = tuple(
                    torch.stack(parts, dim=2) for parts in zip(*encoder_kv)
                )
            cache["clean_length"] = length
        mask = layout.attention_mask[:, :, length:, :]
        for block, prefix in zip(self.model.blocks, decoder_kv):
            if self.context_attention is not None:
                prefix = cache.get("encoder_depth_kv")
            noisy = block(
                noisy,
                noisy_rotary,
                condition,
                mask,
                key_value_prefix=prefix,
                context_attention=self.context_attention,
            )
        return ModelPrediction(self.model.project_hidden(noisy, condition), noisy)
