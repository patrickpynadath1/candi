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
        self.adaLN = config.algo.time_conditioning
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        self.position_rotary_emb = PositionRotary(dim // config.model.n_heads)

        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim, n_heads=config.model.n_heads, dropout=config.model.dropout
                )
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, sigma, get_logits=True):
        x = self.vocab_embed(x)
        if self.adaLN:
            t_cond = F.silu(self.sigma_map(sigma))
        else:
            t_cond = None

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            if get_logits:
                x = self.output_layer(x, c=t_cond)
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

        hidden = self.vocab_embed(x)
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
        return self.vocab_embed(x)

    def forward_with_embeddings(self, x_emb, sigma):
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x_emb)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            x = x_emb
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            x = self.output_layer(x, c=t_cond)

        return x
