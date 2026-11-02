"""Contextual latent autoencoding composed with the existing E2D2 wrapper."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from core.models.layers import (
    DDiTBlock,
    DDiTFinalLayer,
    EmbeddingLayer,
    Rotary,
    TimestepEmbedder,
)
from core.models.components import DiffusionBlocks
from core.models.text_dit import TextDiT
from core.models.kv_cache import append_key_value
from core.modifiers.block_diffusion.layout import make_layout


class GaussianScaleNormalization(nn.Module):
    """Per-position LayerNorm followed by radius sqrt(d), without learned scale."""

    def forward(self, hidden):
        hidden = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        return F.normalize(hidden, dim=-1, eps=1e-8) * math.sqrt(hidden.shape[-1])


class TokenLatentEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, n_blocks=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = EmbeddingLayer(hidden_size, vocab_size)
        self.rotary_emb = Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList(
            DDiTBlock(hidden_size, n_heads, False, dropout=dropout)
            for _ in range(n_blocks)
        )
        self.block_execution = DiffusionBlocks()

    def forward(self, tokens, block_size, cache=None):
        length = tokens.shape[1]
        if cache is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("encoder cache requires eval mode without gradients")
        if cache is not None:
            old = cache.get("tokens")
            if old is not None and (
                length < old.shape[1] or not torch.equal(tokens[:, : old.shape[1]], old)
            ):
                cache.clear()
            old_length = cache.get("length", 0)
            if old_length % block_size and length > old_length:
                cache.clear()
            start = cache.get("length", 0)
        else:
            start = 0
        hidden = self.embedding(tokens[:, start:])
        if length == start:
            return cache.get("hidden", hidden) if cache is not None else hidden
        layout = make_layout(
            length,
            block_size,
            tokens.device,
            batch_size=len(tokens),
            clean_length=length,
            x0_causal=False,
        )
        mask = layout.attention_mask[:, :, start:length, :length]
        with torch.autocast(hidden.device.type, enabled=False):
            rotary = self.rotary_emb(
                hidden, position_ids=layout.position_ids[:, start:length]
            )
        if cache is None:
            return self.block_execution(
                hidden, self.blocks, rotary, attention_mask=mask
            )
        pairs = cache.setdefault("kv", [None] * len(self.blocks))
        for i, block in enumerate(self.blocks):
            hidden, pair = block(
                hidden,
                rotary,
                attention_mask=mask,
                key_value_prefix=pairs[i],
                return_key_value=True,
            )
            pairs[i] = append_key_value(pairs[i], pair)
        cache["hidden"] = torch.cat((cache["hidden"], hidden), 1) if start else hidden
        cache["tokens"], cache["length"] = tokens.clone(), length
        return cache["hidden"]


class LatentTransformer(TextDiT):
    """Hidden-to-hidden/token backbone, with no unused token embedding table."""

    def __init__(
        self,
        output_dim,
        hidden_size=512,
        n_blocks=8,
        n_heads=8,
        dropout=0.1,
        time_conditioning=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rotary_emb = Rotary(hidden_size // n_heads)
        self.sigma_map = TimestepEmbedder(hidden_size) if time_conditioning else None
        self.blocks = nn.ModuleList(
            DDiTBlock(
                hidden_size, n_heads, time_conditioning, hidden_size, dropout=dropout
            )
            for _ in range(n_blocks)
        )
        self.output_layer = DDiTFinalLayer(
            hidden_size, output_dim, hidden_size, time_conditioning
        )

    def condition(self, times):
        return F.silu(self.sigma_map(times)) if self.sigma_map is not None else None


class LatentAutoencoderModel(nn.Module):
    def __init__(self, encoder, denoiser, decoder, normalization, block_size=32):
        super().__init__()
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        widths = [encoder.hidden_size, denoiser.hidden_size, decoder.hidden_size]
        if len(set(widths)) != 1:
            raise ValueError("encoder, denoiser and decoder widths must match")
        if denoiser.output_layer.linear.out_features != encoder.hidden_size:
            raise ValueError("denoiser must predict the latent dimension")
        self.encoder, self.denoiser, self.decoder = encoder, denoiser, decoder
        self.normalization, self.block_size = normalization, block_size
        self.hidden_size = encoder.hidden_size

    def encode(self, tokens, cache=None):
        return self.normalization(self.encoder(tokens, self.block_size, cache=cache))

    def layout(self, state, context, offset=0):
        return make_layout(
            state.shape[1],
            self.block_size,
            state.device,
            batch_size=len(state),
            offset=offset,
            clean_length=context.shape[1],
            x0_causal=False,
        )

    def denoise(self, state, times, context, *, offset=0, cache=None):
        condition = self.denoiser.model.condition(times)
        return self.denoiser(
            context, state, condition, self.layout(state, context, offset), cache=cache
        ).logits

    def decode(self, state, context, *, offset=0, cache=None):
        return self.decoder(
            context, state, None, self.layout(state, context, offset), cache=cache
        ).logits
