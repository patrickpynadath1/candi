import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig

from .blocks import AbsolutePositionalEmbedding, FeedForwardNetwork
from .latent_attention_ntv import LatentAttentionNtv


class ScaleMask(nn.Module):
    def __init__(self,):
        super().__init__()
        self.scale_mlp = nn.Linear(1, 1, bias=True)
        
    def forward(self, x, mask):
        scale = self.scale_mlp(mask.unsqueeze(-1))
        x = x * (scale + 1)
        return x


# have to do some rewiring to adapt to current codebase 
class DecoderTransformerBlock(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.hidden_size = config.model.hidden_size
        self.ff_mult = config.model.ff_mult
        self.dropout = config.model.dropout
        self.config = config
        
        self.latent_attn = LatentAttentionNtv(
            config=config,
            latents_first=False
        )
        self.attn_scale = ScaleMask()
        
        
        self.ffn_latent = FeedForwardNetwork(
            embedding_dim=self.hidden_size,
            mult=self.ff_mult,
            dropout_p=self.dropout
        )
        self.ffn_scale = ScaleMask()
        
    def forward(self, hidden_state_of_embs, hidden_state_of_latents, mask_tokens, mask_latents, mask_of_mask_tokens):
        hidden_state_of_embs_add = self.latent_attn(
            hidden_state_of_embs=hidden_state_of_latents, 
            hidden_state_of_latents=hidden_state_of_embs, 
            mask_tokens=mask_latents, 
            mask_latents=mask_tokens,
        )
        hidden_state_of_embs = hidden_state_of_embs + self.attn_scale(hidden_state_of_embs_add, mask_of_mask_tokens)
        
        hidden_state_of_embs = hidden_state_of_embs + self.ffn_scale(self.ffn_latent(hidden_state_of_embs), mask_of_mask_tokens)
        return hidden_state_of_embs

class DecoderNtv(nn.Module):
    def __init__(self, config: DictConfig, vocab_size):
        super().__init__()
        

        self.config = config 
        self.max_position_embeddings = config.model.k_register_latents 
        self.embedding_dim = config.model.hidden_size
        self.vocab_size = vocab_size 
        self.num_hidden_layers = config.model.n_blocks
        self.num_latents = config.model.k_register_latents
        self.latent_dim = config.model.hidden_size
        self.max_seq_len = config.model.length 

        self.initial_embedding = nn.Parameter(torch.randn(self.embedding_dim) * 0.02)

        
        # positional encodings for encoder latents
        self.positional_emb = AbsolutePositionalEmbedding(
            self.embedding_dim,
            self.max_position_embeddings
        )
        if self.embedding_dim != self.latent_dim:
            self.positional_latent = AbsolutePositionalEmbedding(
                self.latent_dim,
                self.num_latents
            )
        
        # Token embedding table. This can be initialized later from the aux MLM.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.embedding.requires_grad_(False)
        
        # layers
        self.embedding_ffn = FeedForwardNetwork(
            embedding_dim=self.embedding_dim,
            mult=config.model.ff_mult,
            dropout_p=config.model.dropout
        )
        self.layers = nn.ModuleList([
            DecoderTransformerBlock(config=deepcopy(config)) 
            for _ in range(self.num_hidden_layers)
        ])
        
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.scale_embedding = ScaleMask()

    def forward(self, encoder_latents,  return_last_hidden_state=False):
        batch_size = encoder_latents.shape[0]
        
        # Create masks if not provided
        encoder_latents_mask = torch.ones(
            (encoder_latents.shape[0], encoder_latents.shape[1]),
            dtype=encoder_latents.dtype,
            device=encoder_latents.device
        )

        embedding = self.initial_embedding[None, None, :].repeat(batch_size, self.max_seq_len, 1).to(encoder_latents.device)

        tokens_mask = torch.ones(
            (batch_size, embedding.shape[1]),
            dtype=encoder_latents.dtype,
            device=encoder_latents.device
        )

        mask_of_mask_tokens = torch.ones(
            (batch_size, embedding.shape[1]),
            device=encoder_latents.device
        )
        
        hidden_state_of_decoder = self.scale_embedding(self.embedding_ffn(embedding), mask_of_mask_tokens) + self.positional_emb(embedding)
        if self.embedding_dim != self.latent_dim:
            hidden_state_of_encoder_latents = encoder_latents + self.positional_latent(encoder_latents)
        else:
            hidden_state_of_encoder_latents = encoder_latents + self.positional_emb(encoder_latents)

        for layer in self.layers:
            hidden_state_of_decoder = layer(
                hidden_state_of_embs=hidden_state_of_decoder,
                hidden_state_of_latents=hidden_state_of_encoder_latents,
                mask_tokens=tokens_mask,
                mask_latents=encoder_latents_mask,
                mask_of_mask_tokens=mask_of_mask_tokens,
            )
            
        logits = self.lm_head(hidden_state_of_decoder)
        if return_last_hidden_state:
            return logits, hidden_state_of_decoder
        else:
            return logits
        
