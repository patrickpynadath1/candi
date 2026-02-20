import math
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
    DDiTBlockCausal,
    DDiTBlock,
    DDiTFinalLayer,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    CrossAttentionEncoderBlock, 
    CrossAttentionDecoderBlock,
    RMSNorm,
)
# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)



class DIT_Encoder(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)

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
        self.to_latent = CrossAttentionEncoderBlock(dim=dim, 
                                                    n_heads=config.model.n_heads, 
                                                    dropout=config.model.dropout, 
                                                    n_latents=config.model.n_latents,
                                                    adaLN=self.adaLN, 
                                                    cond_dim=cond_dim)
    
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, sigma):
        x = self.vocab_embed(x)
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
        latents = self.to_latent(x, c=t_cond)
        return latents
    
    def get_embedding(self, x):
        return self.vocab_embed(x)

class DIT_Encoder(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        dim = config.hidden_size
        cond_dim = config.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.n_heads)

        blocks = []
        for _ in range(config.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim, n_heads=config.n_heads, dropout=config.dropout
                )
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.dropout,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.to_latent = CrossAttentionEncoderBlock(dim=dim, 
                                                    n_heads=config.n_heads, 
                                                    dropout=config.dropout, 
                                                    n_latents=config.n_latents,
                                                    adaLN=self.adaLN, 
                                                    cond_dim=cond_dim)
    
        self.scale_by_sigma = config.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, sigma, skip_embedding=False):
        if not skip_embedding:
            x = self.vocab_embed(x)
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
        latents = self.to_latent(x, c=t_cond)
        return latents
    
    def get_embedding(self, x):
        return self.vocab_embed(x)
    


class DIT_Decoder(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        dim = config.hidden_size
        cond_dim = config.cond_dim
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.n_heads)

        blocks = []
        for _ in range(config.n_blocks):
            
            block = DDiTBlock(
                dim=dim,
                n_heads=config.n_heads,
                cond_dim=cond_dim,
                adaLN=self.adaLN,
                dropout=config.dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.from_latents = CrossAttentionDecoderBlock(dim=dim, 
                                                    n_heads=config.n_heads, 
                                                    dropout=config.dropout, 
                                                    n_outputs=config.length,
                                                    adaLN=self.adaLN, 
                                                    cond_dim=cond_dim)
        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )
    
        self.scale_by_sigma = config.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, sigma, skip_embedding=False):
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))


        x = self.from_latents(x, c=t_cond)
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            if not skip_embedding:
                x = self.output_layer(x, c=t_cond)
        return x
    

class EncoderDecoderSphere(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.encoder = DIT_Encoder(config, vocab_size)
        self.decoder = DIT_Decoder(config, vocab_size)
        self.spherifying = RMSNorm(config.hidden_size)

    def forward(self, x, sigma):
        latents = self.encoder(x, sigma)
        latents = self.spherifying(latents)
        return self.decoder(latents, sigma), latents
    
    def encode(self, x, sigma, skip_embedding=False): 
        return self.spherifying(self.encoder(x, sigma, skip_embedding=skip_embedding))

    def decode(self, latents, sigma, spherify_first=True, skip_embedding=False):
        if spherify_first:
            latents = self.spherifying(latents)
        return self.decoder(latents, sigma, skip_embedding=skip_embedding)
    
    def spherify(self, latents):
        return self.spherifying(latents)

    

