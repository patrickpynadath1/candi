"""Transformer split around a normalized latent denoising boundary."""

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DDiTBlock, DDiTFinalLayer, EmbeddingLayer, Rotary, TimestepEmbedder
from .registry import models


@models.register("latent_split_dit")
class LatentSplitDIT(nn.Module):
    """A token encoder followed by a time-conditioned latent decoder."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        n_blocks = config.model.n_blocks
        ratio = float(config.model.encoder_ratio)
        if n_blocks < 2:
            raise ValueError("latent_split_dit requires at least two transformer blocks")
        if not 0 < ratio < 1:
            raise ValueError("model.encoder_ratio must be strictly between 0 and 1")
        self.encoder_blocks = int(round(n_blocks * ratio))
        self.encoder_blocks = max(1, min(n_blocks - 1, self.encoder_blocks))
        self.decoder_blocks = n_blocks - self.encoder_blocks

        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=index >= self.encoder_blocks,
                    dropout=config.model.dropout,
                )
                for index in range(n_blocks)
            ]
        )
        # Non-affine normalization prevents the encoder from learning scale back in.
        self.latent_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.posterior_mean = nn.Linear(dim, dim)
        self.posterior_log_variance = nn.Linear(dim, dim)
        nn.init.eye_(self.posterior_mean.weight)
        nn.init.zeros_(self.posterior_mean.bias)
        nn.init.zeros_(self.posterior_log_variance.weight)
        nn.init.zeros_(self.posterior_log_variance.bias)
        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=True,
        )
        if getattr(config.model, "tie_word_embeddings", False):
            self.output_layer.linear.weight = self.vocab_embed.embedding

    def embed_tokens(self, tokens):
        return self.vocab_embed(tokens)

    def encode_features(self, embeddings):
        hidden = embeddings
        rotary = self.rotary_emb(hidden)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
            for block in self.blocks[: self.encoder_blocks]:
                hidden = block(hidden, rotary, c=None)
        return self.latent_norm(hidden.float()).to(hidden.dtype)

    def posterior_from_embeddings(self, embeddings):
        features = self.encode_features(embeddings)
        mean = self.posterior_mean(features)
        log_variance = self.posterior_log_variance(features).clamp(-20, 10)
        return mean, log_variance

    def posterior_from_tokens(self, tokens):
        return self.posterior_from_embeddings(self.embed_tokens(tokens))

    @staticmethod
    def reparameterize(mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + standard_deviation * torch.randn_like(standard_deviation)

    def encode_embeddings(self, embeddings):
        """Return the posterior mean for deterministic inference feedback."""
        mean, _ = self.posterior_from_embeddings(embeddings)
        return mean

    def encode_tokens(self, tokens):
        mean, _ = self.posterior_from_tokens(tokens)
        return mean

    def decode_latents(self, latents, time):
        condition = F.silu(self.sigma_map(time))
        hidden = latents
        rotary = self.rotary_emb(hidden)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
            for block in self.blocks[self.encoder_blocks :]:
                hidden = block(hidden, rotary, c=condition)
            return self.output_layer(hidden, c=condition)

    def feedback_embeddings(self, logits, mode="soft", temperature=1.0):
        if temperature <= 0:
            raise ValueError("feedback temperature must be positive")
        scaled = logits.float() / temperature
        if mode == "soft":
            probabilities = scaled.softmax(-1)
            return torch.einsum(
                "blv,vd->bld", probabilities, self.vocab_embed.embedding.float()
            ).to(logits.dtype)
        if mode == "greedy":
            return self.embed_tokens(scaled.argmax(-1))
        if mode == "sample":
            probabilities = scaled.softmax(-1)
            tokens = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]), 1
            ).reshape(probabilities.shape[:-1])
            return self.embed_tokens(tokens)
        raise ValueError(f"unknown latent feedback mode: {mode}")
