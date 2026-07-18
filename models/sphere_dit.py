"""DiT operating directly on unit-hypersphere token embeddings."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.spherical_utils import sphere_normalize
from .common import DDiTBlock, DDiTFinalLayer, Rotary, TimestepEmbedder
from .registry import models
from .sphere_checkpoint import load_compatible_checkpoint


@models.register("sphere_dit")
class SphereDiT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        dim, cond_dim = config.model.hidden_size, config.model.cond_dim
        self.embed_dim, self.eps = dim, getattr(config.model, "eps", 1e-6)
        self.sphere_embed = nn.Embedding(vocab_size, dim)
        init = getattr(config.model, "init", "ngpt")
        if init == "pretrained":
            nn.init.zeros_(self.sphere_embed.weight)
        elif init in {"ngpt", "random"}:
            std = 1 / math.sqrt(dim) if init == "ngpt" else 0.02
            nn.init.normal_(self.sphere_embed.weight, std=std)
        else:
            raise ValueError(f"unknown sphere embedding initialization: {init}")
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim,
                    config.model.n_heads,
                    adaLN=True,
                    cond_dim=cond_dim,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.learn_temperature = getattr(
            config.model, "learn_temperature_scaling", False
        )
        self.output_layer = DDiTFinalLayer(
            dim, vocab_size + int(self.learn_temperature), cond_dim, True
        )
        path = getattr(config.model, "pretrained_ckpt_path", None)
        if path is not None:
            load_compatible_checkpoint(self, path)

    def get_sphere_embeddings(self, tokens):
        return sphere_normalize(self.sphere_embed(tokens), self.eps)

    @torch.no_grad()
    def renormalize_weights(self):
        self.sphere_embed.weight.copy_(
            sphere_normalize(self.sphere_embed.weight, self.eps)
        )

    def forward(self, x, sigma):
        cond = F.silu(self.sigma_map(sigma))
        rotary = self.rotary_emb(x)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            for block in self.blocks:
                x = block(x, rotary, c=cond)
            logits = self.output_layer(x, c=cond)
        if self.learn_temperature:
            logits, raw_temp = logits[..., :-1], logits[..., -1:]
            logits = logits / (1.000001 + torch.tanh(raw_temp) * 0.999999)
        return logits
