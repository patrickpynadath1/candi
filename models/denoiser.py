# Assumes that the vocab size accounts for the mask token 
from models.common import DDiTBlock, Rotary, TimestepEmbedder, bias_dropout_add_scale_fused_inference, bias_dropout_add_scale_fused_train
from models.registry import models


import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F


@models.register("denoiser")
class Denoiser(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, num_decoder_stacks = None):

        if num_decoder_stacks is None:
            num_decoder_stacks = config.model.n_blocks // 2
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = True
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        self.dim=dim
        cond_dim = config.model.cond_dim
        self.mask_index = vocab_size-1
        self.k_register_latents = config.model.k_register_latents


        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        blocks = []

        self.final_normalization = nn.LayerNorm(dim)
        for _ in range(num_decoder_stacks):
            block = DDiTBlock(
                dim=dim,
                n_heads=config.model.n_heads,
                cond_dim=cond_dim,
                adaLN=self.adaLN,
                dropout=config.model.dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        # self.output_layer = DDiTFinalLayer(
        #     hidden_size=dim,
        #     out_channels=vocab_size,
        #     cond_dim=cond_dim,
        #     adaLN=self.adaLN,
        # )



    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def _get_mask_embedding(self):
        return self.vocab_embed.embedding[self.mask_index]

    # clean mask: indicates which positions are clean with 1, which are corrupted with 0
    # def forward(self, x, discrete_noise):
    def forward(self,
                input_latent,
                sigma,
                generation_mode=False,
                **kwargs):

        t_cond = F.silu(self.sigma_map(sigma))
        x = input_latent
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)

            x = self.final_normalization(x)
        return x



    def get_embedding(self, x):
        return self.vocab_embed(x)

    def get_mask_embed(self):
        return self.vocab_embed.embedding[self.mask_index]