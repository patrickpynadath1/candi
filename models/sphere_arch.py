"""nGPT-inspired normalized architecture used by S-FLM."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.spherical_utils import sphere_normalize
from .common import Rotary, TimestepEmbedder, split_and_apply_rotary_pos_emb
from .registry import models
from .sphere_checkpoint import load_compatible_checkpoint


class SphereArchBlock(nn.Module):
    def __init__(
        self, dim, heads, cond_dim, dropout, mlp_type="gelu", time_alpha=True, eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads
        self.eps = eps
        self.dropout = dropout
        self.mlp_type = mlp_type
        self.time_alpha = time_alpha
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.fc = nn.Linear(dim, dim * 4 * (2 if mlp_type == "swiglu" else 1))
        self.proj = nn.Linear(dim * 4, dim)
        self.qk_scale = nn.Parameter(torch.ones(dim) / math.sqrt(dim))
        self.attn_alpha = nn.Parameter(torch.ones(dim) / math.sqrt(dim))
        self.mlp_alpha = nn.Parameter(torch.ones(dim) / math.sqrt(dim))
        self.alpha_mod = nn.Linear(cond_dim, 2 * dim) if time_alpha else None
        if self.alpha_mod is not None:
            nn.init.zeros_(self.alpha_mod.weight)
            nn.init.zeros_(self.alpha_mod.bias)

    def forward(self, h, rotary, cond):
        batch, length, dim = h.shape
        q = self.q(h).view(batch, length, self.heads, self.dk)
        k = self.k(h).view(batch, length, self.heads, self.dk)
        v = self.v(h).view(batch, length, self.heads, self.dk)
        q, k, v = split_and_apply_rotary_pos_emb(torch.stack((q, k, v), 2), rotary)
        scale = (self.qk_scale * math.sqrt(self.dim)).view(1, 1, self.heads, self.dk)
        q = sphere_normalize(q, self.eps) * scale
        k = sphere_normalize(k, self.eps) * scale
        att = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=math.sqrt(self.dk),
            )
            .transpose(1, 2)
            .reshape(batch, length, dim)
        )
        da, dm = (
            self.alpha_mod(cond).chunk(2, -1) if self.alpha_mod is not None else (0, 0)
        )
        normalized_h = sphere_normalize(h, self.eps)
        normalized_att = sphere_normalize(self.o(att), self.eps)
        alpha_att = (self.attn_alpha * (0.05 * math.sqrt(self.dim)) + da).abs()
        h = sphere_normalize(
            normalized_h + alpha_att.unsqueeze(1) * (normalized_att - normalized_h),
            self.eps,
        )
        z = self.fc(h)
        if self.mlp_type == "swiglu":
            u, g = z.chunk(2, -1)
            z = u * F.silu(g)
        else:
            z = F.gelu(z)
        normalized_h = sphere_normalize(h, self.eps)
        normalized_mlp = sphere_normalize(self.proj(z), self.eps)
        alpha_mlp = (self.mlp_alpha * (0.05 * math.sqrt(self.dim)) + dm).abs()
        return sphere_normalize(
            normalized_h + alpha_mlp.unsqueeze(1) * (normalized_mlp - normalized_h),
            self.eps,
        )


@models.register("sphere_arch")
class SphereArch(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        d = config.model.hidden_size
        c = config.model.cond_dim
        self.embed_dim = d
        self.eps = getattr(config.model, "eps", 1e-6)
        self.sphere_embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.sphere_embed.weight, std=1 / math.sqrt(d))
        self.time = TimestepEmbedder(c)
        self.use_time_token = getattr(config.model, "use_time_token", False)
        self.normalize_input_embed = getattr(
            config.model, "normalize_input_embed", False
        )
        self.time_token = nn.Linear(c, d, bias=False) if self.use_time_token else None
        self.rotary = Rotary(d // config.model.n_heads)
        self.blocks = nn.ModuleList(
            [
                SphereArchBlock(
                    d,
                    config.model.n_heads,
                    c,
                    config.model.dropout,
                    getattr(config.model, "mlp_type", "gelu"),
                    getattr(config.model, "use_time_alpha", True),
                    self.eps,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.out_scale = nn.Parameter(torch.ones(vocab_size) / math.sqrt(d))
        path = getattr(config.model, "pretrained_ckpt_path", None)
        if path is not None:
            load_compatible_checkpoint(self, path)
        self.renormalize_weights()

    def get_sphere_embeddings(self, tokens):
        return sphere_normalize(self.sphere_embed(tokens), self.eps)

    @torch.no_grad()
    def renormalize_weights(self):
        for name, p in self.named_parameters():
            if name == "sphere_embed.weight" and not self.normalize_input_embed:
                continue
            if p.ndim >= 2:
                p.copy_(sphere_normalize(p, self.eps))

    def forward(self, x, sigma):
        cond = F.silu(self.time(sigma))
        if self.time_token is not None:
            token = sphere_normalize(self.time_token(cond), self.eps).unsqueeze(1)
            x = torch.cat((token, x), dim=1)
        rotary = self.rotary(x)
        for block in self.blocks:
            x = block(x, rotary, cond)
        if self.time_token is not None:
            x = x[:, 1:]
        return self.head(x) * (self.out_scale * math.sqrt(self.embed_dim)).abs()
