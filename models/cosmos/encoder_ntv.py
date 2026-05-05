import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig

from .blocks import AbsolutePositionalEmbedding, RMSNorm, FeedForwardNetwork
from .latent_attention_ntv import LatentAttentionNtv


class EncoderTransformerBlockNtv(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.cfg = config
        
        self.latent_attn = LatentAttentionNtv(
            config=config,
            latents_first=True
        )
        
        self.ffn_latent = FeedForwardNetwork(
            embedding_dim=config.model.hidden_size, 
            mult=config.model.ff_mult, 
            dropout_p=config.model.dropout, 
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


class EncoderNtv(nn.Module):
    def __init__(self, config: DictConfig, vocab_size):
        super().__init__()
        
        self.config=config
        self.input_embedding = nn.Embedding(vocab_size, config.model.hidden_size)
        self.num_latents = config.model.k_register_latents
        self.latent_dim = config.model.hidden_size
        self.embedding_dim = config.model.hidden_size
        self.vocab_size = vocab_size
        self.num_hidden_layers = config.model.n_blocks

        self.max_position_embeddings = self.num_latents
        self.max_seq_len = self.max_position_embeddings

        # latents
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.normal_(self.latents, std=.02)
        self.latents.requires_grad_(True)
        
        # positional encodings
        self.positional_emb = AbsolutePositionalEmbedding(
            self.embedding_dim,
            self.max_position_embeddings
        )

        # layers
        self.latent_norm = RMSNorm(
            self.latent_dim,
            eps=config.model.layer_eps
        )
        
        self.latent_layers = nn.ModuleList([
            EncoderTransformerBlockNtv(config=deepcopy(config)) 
            for _ in range(self.num_hidden_layers)
        ])
        
        self.output_norm = RMSNorm(
            self.latent_dim,
            eps=config.model.layer_eps
        )

    def forward(
        self,
        input_tokens,
        attn_mask=None,
        **kwargs,
    ):
        token_embeddings = self.input_embedding(input_tokens)


        if attn_mask is None:
            attn_mask = torch.ones_like(input_tokens, dtype=torch.long)
            
        mask_latents = torch.ones(
            (token_embeddings.shape[0], self.num_latents),
            dtype=attn_mask.dtype,
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
                mask_tokens=attn_mask, 
                mask_latents=mask_latents,
            )
        
        hidden_state_of_latents = self.output_norm(hidden_state_of_latents)
        return hidden_state_of_latents
