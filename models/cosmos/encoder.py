import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig

from .blocks import AbsolutePositionalEmbedding, RMSNorm, FeedForwardNetwork
from .latent_attention import LatentAttention


class EncoderTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        self.latent_attn = LatentAttention(
            cfg=cfg,
            latents_first=True
        )
        
        self.ffn_latent = FeedForwardNetwork(
            embedding_dim=cfg.latent.dim,
            mult=cfg.hidden.ff_mult,
            dropout_p=cfg.hidden.dropout
        )
        
    def forward(self, hidden_state_of_embs, hidden_state_of_latents, mask_tokens, mask_latents):
        hidden_state_of_latents_add = self.latent_attn(
            hidden_state_of_embs=hidden_state_of_embs, 
            hidden_state_of_latents=hidden_state_of_latents, 
            mask_tokens=mask_tokens, 
            mask_latents=mask_latents,
        )
        hidden_state_of_latents = hidden_state_of_latents + hidden_state_of_latents_add
        
        hidden_state_of_latents = hidden_state_of_latents + self.ffn_latent(hidden_state_of_latents)
        return hidden_state_of_latents


class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        self.num_latents = cfg.latent.num_latents
        self.latent_dim = cfg.latent.dim
        self.embedding_dim = cfg.embedding.dim
        self.vocab_size = cfg.tokens.vocab_size
        self.num_hidden_layers = cfg.hidden.num_layers

        self.max_position_embeddings = max(cfg.embedding.max_position_embeddings, cfg.latent.num_latents)
        self.max_seq_len = self.max_position_embeddings

        # latents
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.normal_(self.latents, std=cfg.embedding.initializer_range)
        self.latents.requires_grad_(True)
        
        # positional encodings
        self.positional_emb = AbsolutePositionalEmbedding(
            self.embedding_dim,
            self.max_position_embeddings
        )

        # layers
        self.latent_norm = RMSNorm(
            self.latent_dim,
            eps=cfg.normalization.layer_eps
        )
        
        self.latent_layers = nn.ModuleList([
            EncoderTransformerBlock(cfg=deepcopy(cfg)) 
            for _ in range(self.num_hidden_layers)
        ])
        
        self.output_norm = RMSNorm(
            self.latent_dim,
            eps=cfg.normalization.layer_eps
        )

    def forward(
        self,
        input_tokens=None,
        sigma=None,
        skip_embedding=False,
        token_ids=None,
        mask_tokens=None,
        token_embeddings=None,
        **kwargs,
    ):
        del sigma, kwargs

        if token_ids is None:
            token_ids = input_tokens

        if skip_embedding:
            token_embeddings = input_tokens

        if token_embeddings is None:
            raise ValueError(
                "Cosmos encoder expects external hidden representations. "
                "Pass them via `token_embeddings=...` or call with `skip_embedding=True`."
            )

        if mask_tokens is None:
            if token_ids is not None:
                mask_tokens = torch.ones_like(token_ids)
            else:
                mask_tokens = torch.ones(
                    token_embeddings.shape[:2],
                    dtype=torch.long,
                    device=token_embeddings.device,
                )
            
        mask_latents = torch.ones(
            (token_embeddings.shape[0], self.num_latents),
            dtype=mask_tokens.dtype,
            device=token_embeddings.device
        )

        hidden_state_of_embs = token_embeddings + self.positional_emb(token_embeddings)
        
        hidden_state_of_latents = self.latents.view(1, self.num_latents, self.latent_dim).repeat(
            token_embeddings.shape[0], 1, 1
        )
        hidden_state_of_latents = self.latent_norm(hidden_state_of_latents)
        
        for layer in self.latent_layers:
            hidden_state_of_latents = layer(
                hidden_state_of_embs=hidden_state_of_embs, 
                hidden_state_of_latents=hidden_state_of_latents, 
                mask_tokens=mask_tokens, 
                mask_latents=mask_latents,
            )
        
        hidden_state_of_latents = self.output_norm(hidden_state_of_latents)
        return hidden_state_of_latents
