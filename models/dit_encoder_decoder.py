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
        self.causal = False
        self.adaLN = False
        self.config = config
        self.vocab_size = vocab_size
        dim = config.hidden_size
        cond_dim = config.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
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
    

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, skip_embedding=False):
        if not skip_embedding:
            x = self.vocab_embed(x)
        t_cond = None

        rotary_cos_sin = self.rotary_emb(x)

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
        self.adaLN = False
        self.config = config
        self.vocab_size = vocab_size
        dim = config.hidden_size
        cond_dim = config.cond_dim
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

    def forward(self, x, skip_embedding=False):
        t_cond = None
        x = self.from_latents(x, c=t_cond)
        rotary_cos_sin = self.rotary_emb(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
        if not skip_embedding:
            x = self.output_layer(x, c=t_cond)
        return x
    

class EncoderDecoder(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.encoder = DIT_Encoder(config, vocab_size)
        self.decoder = DIT_Decoder(config, vocab_size)

    def forward(self, x):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            latents = self.encoder(x)
            return self.decoder(latents), latents
    
    def encode(self, x, skip_embedding=False): 
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return self.encoder(x, skip_embedding=skip_embedding)

    def decode(self, latents, skip_embedding=False):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return self.decoder(latents, skip_embedding=skip_embedding)
