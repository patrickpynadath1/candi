"""Modern DiT denoiser used by RePlaid."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DDiTBlock, DDiTFinalLayer, Rotary, TimestepEmbedder
from .registry import models


@models.register("replaid_dit")
class RePlaidDIT(nn.Module):
    """Low-dimensional Gaussian input with categorical token prediction."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        embed_dim = config.algo.embedding_dim
        hidden_size = config.model.hidden_size
        cond_dim = config.model.cond_dim

        self.token_embedding = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.noisy_projection = nn.Linear(embed_dim, hidden_size, bias=False)
        self.self_condition_projection = nn.Linear(embed_dim, hidden_size, bias=False)
        self.time_input_projection = nn.Linear(cond_dim, hidden_size, bias=False)
        self.time_embedding = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // config.model.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=hidden_size,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=True,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.output_layer = DDiTFinalLayer(
            hidden_size=hidden_size,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=True,
        )
        self.project_embeddings_()

    @property
    def normalized_embedding(self):
        return F.normalize(self.token_embedding.float(), dim=-1)

    @torch.no_grad()
    def project_embeddings_(self):
        self.token_embedding.copy_(F.normalize(self.token_embedding, dim=-1))

    def embed_tokens(self, tokens, detach_embedding_mask=None):
        if tokens.ndim == 2:
            return F.embedding(tokens, self.normalized_embedding)
        embedded = torch.einsum(
            "blv,ve->ble", tokens.float(), self.normalized_embedding
        )
        if detach_embedding_mask is not None:
            detached = torch.einsum(
                "blv,ve->ble",
                tokens.float(),
                self.normalized_embedding.detach(),
            )
            embedded = torch.where(
                detach_embedding_mask[:, None, None], detached, embedded
            )
        return embedded

    def forward(
        self,
        z_t,
        gamma_t,
        self_condition=None,
        output_prior=True,
        detach_embedding_mask=None,
    ):
        if self_condition is None:
            self_condition = torch.zeros_like(z_t)
        condition = F.silu(self.time_embedding(gamma_t.float()))
        hidden = (
            self.noisy_projection(z_t.float())
            + self.self_condition_projection(self_condition.float())
            + self.time_input_projection(condition)[:, None, :]
        )
        rotary = self.rotary_emb(hidden)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda
        ):
            for block in self.blocks:
                hidden = block(hidden, rotary, c=condition)
            logits = self.output_layer(hidden, c=condition)
        logits = logits.float()
        if output_prior:
            alpha_sq = torch.sigmoid(-gamma_t.float())
            sigma_sq = torch.sigmoid(gamma_t.float())
            # -||z-alpha*e_v||^2/(2*sigma^2) differs across vocabulary
            # only through (alpha/sigma^2) z·e_v because every e_v has unit norm.
            # Dropping the common terms avoids a [B,L,V,d_e] temporary.
            prior = torch.einsum(
                "ble,ve->blv", z_t.float(), self.normalized_embedding
            )
            if detach_embedding_mask is not None:
                detached_prior = torch.einsum(
                    "ble,ve->blv",
                    z_t.float(),
                    self.normalized_embedding.detach(),
                )
                prior = torch.where(
                    detach_embedding_mask[:, None, None], detached_prior, prior
                )
            scale = alpha_sq.sqrt() / sigma_sq.clamp_min(1e-12)
            logits = logits + scale[:, None, None] * prior
        return F.log_softmax(logits, dim=-1)
