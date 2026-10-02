"""Causal DiT with explicit-position RoPE and preallocated inference KV caches."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .layers import DDiTBlock, DDiTFinalLayer, EmbeddingLayer, rotate_half


@dataclass
class LayerKVCache:
    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0

    def append(self, keys, values):
        end = self.length + keys.shape[1]
        if (
            keys.shape != values.shape
            or keys.shape[0] != self.keys.shape[0]
            or keys.shape[2:] != self.keys.shape[2:]
        ):
            raise ValueError("KV cache shapes do not match")
        if end > self.keys.shape[1]:
            raise ValueError("KV cache capacity exceeded")
        if keys.device != self.keys.device or keys.dtype != self.keys.dtype:
            raise ValueError("KV cache device/dtype must match attention")
        self.keys[:, self.length : end].copy_(keys)
        self.values[:, self.length : end].copy_(values)
        self.length = end
        return self.keys[:, :end], self.values[:, :end]


@dataclass
class TransformerKVCache:
    layers: list[LayerKVCache]

    @property
    def length(self):
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise ValueError("KV cache layer lengths disagree")
        return next(iter(lengths))


class CausalBlock(DDiTBlock):
    def __init__(self, dim, n_heads, dropout):
        super().__init__(dim, n_heads, False, dropout=dropout)

    def forward(self, hidden, rotary, cache=None, query_rotary=None):
        batch, length, dim = hidden.shape
        qkv = self.attn_qkv(self.norm1(hidden)).reshape(
            batch, length, 3, self.n_heads, dim // self.n_heads
        )
        query, key, value = qkv.unbind(2)
        cos, sin = (x.to(query.dtype) for x in rotary)
        query_cos, query_sin = (
            (cos, sin)
            if query_rotary is None
            else (x.to(query.dtype) for x in query_rotary)
        )
        query = query * query_cos + rotate_half(query) * query_sin
        key = key * cos + rotate_half(key) * sin
        attention_mask = None
        if cache is not None:
            offset = cache.length
            key, value = cache.append(key, value)
            # SDPA's is_causal=True aligns rectangular masks at the upper left;
            # incremental/chunked decoding needs the query's absolute offset.
            attention_mask = (
                torch.arange(key.shape[1], device=hidden.device)[None, :]
                <= offset + torch.arange(length, device=hidden.device)[:, None]
            )
        attended = (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
                is_causal=cache is None,
                dropout_p=0.0,
            )
            .transpose(1, 2)
            .reshape(batch, length, dim)
        )
        hidden = hidden + F.dropout(
            self.attn_out(attended), self.dropout, self.training
        )
        return hidden + F.dropout(
            self.mlp(self.norm2(hidden)), self.dropout, self.training
        )


class CausalDiT(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        n_blocks=8,
        n_heads=8,
        dropout=0.1,
        tie_word_embeddings=False,
    ):
        super().__init__()
        if n_blocks < 1 or hidden_size % n_heads or (hidden_size // n_heads) % 2:
            raise ValueError(
                "require blocks and even integral attention head dimensions"
            )
        self.vocab_size, self.n_heads = vocab_size, n_heads
        self.head_dim = hidden_size // n_heads
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.register_buffer(
            "inv_freq",
            1 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)),
        )
        self.blocks = nn.ModuleList(
            [CausalBlock(hidden_size, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.output_layer = DDiTFinalLayer(hidden_size, vocab_size, None, False)
        if tie_word_embeddings:
            self.output_layer.linear.weight = self.vocab_embed.embedding

    def allocate_kv_cache(self, batch_size, capacity, *, device=None, dtype=None):
        if batch_size <= 0 or capacity <= 0:
            raise ValueError("cache batch and capacity must be positive")
        device = self.vocab_embed.embedding.device if device is None else device
        device_type = torch.device(device).type
        if dtype is None:
            dtype = (
                torch.get_autocast_dtype(device_type)
                if torch.is_autocast_enabled(device_type)
                else self.vocab_embed.embedding.dtype
            )
        shape = (batch_size, capacity, self.n_heads, self.head_dim)
        return TransformerKVCache(
            [
                LayerKVCache(
                    torch.empty(shape, device=device, dtype=dtype),
                    torch.empty(shape, device=device, dtype=dtype),
                )
                for _ in self.blocks
            ]
        )

    def forward(
        self, tokens, cache=None, *, key_position_ids=None, query_position_ids=None
    ):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("expected nonempty [batch, sequence] tokens")
        if cache is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("KV caching requires eval mode and disabled gradients")
        if cache is not None and len(cache.layers) != len(self.blocks):
            raise ValueError("KV cache layer count does not match model")
        offset = 0 if cache is None else cache.length
        if cache is not None and any(
            offset + tokens.shape[1] > layer.keys.shape[1] for layer in cache.layers
        ):
            raise ValueError("KV cache capacity exceeded")
        with torch.autocast(tokens.device.type, enabled=False):
            hidden = self.vocab_embed(tokens)
            positions = torch.arange(
                offset, offset + tokens.shape[1], device=tokens.device
            )[None]
            key_positions = positions if key_position_ids is None else key_position_ids
            query_positions = (
                key_positions if query_position_ids is None else query_position_ids
            )

            def make_rotary(position_ids):
                if (
                    position_ids.shape not in (tokens.shape, (1, tokens.shape[1]))
                    or position_ids.device != tokens.device
                ):
                    raise ValueError("position IDs must match token shape and device")
                if (
                    position_ids.dtype not in (torch.int32, torch.int64)
                    or (position_ids < 0).any()
                ):
                    raise ValueError("position IDs must be nonnegative integers")
                angles = position_ids.float()[..., None] * self.inv_freq.float()
                angles = torch.cat((angles, angles), -1)[:, :, None, :]
                return angles.cos(), angles.sin()

            rotary = make_rotary(key_positions)
            query_rotary = (
                rotary
                if query_positions is key_positions
                else make_rotary(query_positions)
            )
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                rotary,
                None if cache is None else cache.layers[index],
                query_rotary,
            )
        return self.output_layer(hidden, None)
