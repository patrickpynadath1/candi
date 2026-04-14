# Assumes that the vocab size accounts for the mask token 
from models.common import DDiTBlock, DDiTFinalLayer, EmbeddingLayer, Rotary, TimestepEmbedder, bias_dropout_add_scale_fused_inference, bias_dropout_add_scale_fused_train
from models.registry import models


import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F


@models.register("decoder")
class Decoder(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, num_decoder_stacks):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = False
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
        self.input_pos_blanks = EmbeddingLayer(dim, config.model.length)
        self.register_latent_bias = nn.Parameter(torch.zeros(self.k_register_latents, dim))

        # self.final_normalization = nn.LayerNorm(dim)
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
        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )



    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def _get_mask_embedding(self):
        return self.vocab_embed.embedding[self.mask_index]

    def on_after_backward(self):
        for name, param in self.backbone.named_parameters():
            if param.grad is None:
                print(f"Unused: {name}")


    # clean mask: indicates which positions are clean with 1, which are corrupted with 0
    # def forward(self, x, discrete_noise):
    def forward(self,
                input_latent,
                sigma,
                generation_mode=False,
                **kwargs):

        t_cond = None
        if generation_mode:
            x = input_latent
        else:
            input_toks = torch.arange(self.config.model.length, device=input_latent.device).repeat(input_latent.shape[0], 1)
            x = self.input_pos_blanks(input_toks)
            input_latent = input_latent + self.register_latent_bias[None, :, :]
            x = torch.concat([input_latent, x], dim=1)
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)

            logits = self.output_layer(x[:, self.k_register_latents:, :], c=t_cond)
            return logits



    def get_embedding(self, x):
        return self.vocab_embed(x)

    def get_mask_embed(self):
        return self.vocab_embed.embedding[self.mask_index]