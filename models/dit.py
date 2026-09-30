import math
import pdb
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import (
    EmbeddingLayer,
    TimestepEmbedder,
    Rotary,
    PositionRotary,
    LayerKVCache,
    TransformerKVCache,
    DDiTBlockCausal,
    DDiTBlock,
    DDiTFinalLayer,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
)
from .registry import models

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@models.register("dit")
class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        # Older S-FLM checkpoints explicitly record whether AdaLN was used.
        # Fall back to time_conditioning for native configs created before the
        # explicit flag was introduced.
        self.adaLN = getattr(config.algo, "adaLN", config.algo.time_conditioning)
        self.config = config
        self.normalize_embeddings = bool(
            getattr(config.model, "normalize_embeddings", False)
        )
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        if not self.causal or self.adaLN:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        self.position_rotary_emb = PositionRotary(dim // config.model.n_heads)

        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    dropout=config.model.dropout,
                    cond_dim=cond_dim if self.adaLN else None,
                )
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout,
                    exact_sflm_attention=getattr(
                        config.model, "sflm_compat", False
                    ),
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )
        if getattr(config.model, "tie_word_embeddings", False):
            self.output_layer.linear.weight = self.vocab_embed.embedding
        self.scale_by_sigma = config.model.scale_by_sigma
        self.output_dropout = nn.Dropout(getattr(config.model, "output_dropout", 0.0))

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, sigma, get_logits=True):
        x = self.get_embedding(x)
        if self.adaLN:
            t_cond = F.silu(self.sigma_map(sigma))
            if getattr(self.config.algo, "paper_time_conditioning", False) and not self.config.algo.time_conditioning:
                t_cond = torch.zeros_like(t_cond)
        else:
            t_cond = None

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available() and not getattr(self.config.model, "full_precision", False)
        ):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            if get_logits:
                x = self.output_layer(self.output_dropout(x), c=t_cond)
        return x

    def allocate_kv_cache(self, batch_size, capacity, *, device=None, dtype=None):
        """Allocate an inference cache for every causal transformer block."""
        if not self.causal:
            raise ValueError("KV caching is only supported by causal DiT models")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.device(device).type == "cuda"
                else parameter.dtype
            )
        head_dim = self.config.model.hidden_size // self.config.model.n_heads
        return TransformerKVCache(
            [
                LayerKVCache.allocate(
                    batch_size,
                    capacity,
                    self.config.model.n_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in self.blocks
            ]
        )

    def forward_positioned(
        self,
        x,
        *,
        key_position_ids,
        query_position_ids=None,
        cache=None,
        get_logits=True,
    ):
        """Run causal DiT with explicit key/query positions and optional caching."""
        if not self.causal:
            raise ValueError("explicit positioned inference requires a causal DiT")
        if query_position_ids is None:
            query_position_ids = key_position_ids
        if x.shape != key_position_ids.shape or x.shape != query_position_ids.shape:
            raise ValueError(
                "tokens and position IDs must have matching [batch, sequence] shapes"
            )
        if cache is not None and len(cache.layers) != len(self.blocks):
            raise ValueError("KV cache layer count does not match the model")

        hidden = self.get_embedding(x)
        query_rotary = self.position_rotary_emb(query_position_ids)
        key_rotary = self.position_rotary_emb(key_position_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
            for index, block in enumerate(self.blocks):
                layer_cache = None if cache is None else cache.layers[index]
                hidden = block.forward_positioned(
                    hidden, query_rotary, key_rotary, layer_cache
                )
            if get_logits:
                hidden = self.output_layer(hidden, c=None)
        return hidden, cache

    def get_logits(self, h, sigma):
        if sigma is not None:
            sigma = F.silu(self.sigma_map(sigma))
        return self.output_layer(h, c=sigma)

    def get_embedding(self, x):
        embedding = self.vocab_embed(x)
        if self.normalize_embeddings:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

    def forward_with_embeddings(self, x_emb, sigma):
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x_emb)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available() and not getattr(self.config.model, "full_precision", False)
        ):
            x = x_emb
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            x = self.output_layer(self.output_dropout(x), c=t_cond)

        return x

    def forward_masked_embeddings(
        self, x_emb, attention_mask, position_ids, noise_times=None
    ):
        """Run embedded tokens with explicit attention and semantic positions."""
        if not self.causal:
            raise ValueError("masked embedding attention requires a causal DiT")
        if x_emb.shape[:2] != position_ids.shape:
            raise ValueError("embeddings and position IDs must have matching shapes")
        if attention_mask.shape != (x_emb.shape[1], x_emb.shape[1]):
            raise ValueError("attention_mask must be [sequence, sequence]")
        rotary_cos_sin = self.position_rotary_emb(position_ids)
        # PositionRotary is batch-aware, while the older packed helper expects
        # the same position sequence for every example. Block diffusion uses
        # identical semantic positions across the batch, so use the first row.
        rotary_cos_sin = tuple(value[:1] for value in rotary_cos_sin)
        condition = anchor = None
        if self.adaLN:
            if noise_times is None or noise_times.shape != x_emb.shape[:2]:
                raise ValueError("tokenwise noise_times must match [batch, sequence]")
            flat_times = noise_times.reshape(-1)
            condition = F.silu(self.sigma_map(flat_times)).reshape(
                *noise_times.shape, -1
            )
            anchor = F.silu(self.sigma_map(torch.zeros_like(flat_times))).reshape(
                *noise_times.shape, -1
            )
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            hidden = x_emb
            for block in self.blocks:
                hidden = block.forward_masked(
                    hidden, rotary_cos_sin, attention_mask, condition, anchor
                )
            hidden = self.output_layer(hidden, c=condition, c_anchor=anchor)
        return hidden
