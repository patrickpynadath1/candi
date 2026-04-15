# Assumes that the vocab size accounts for the mask token 
from models.common import DDiTBlock, EmbeddingLayer, Rotary, TimestepEmbedder, bias_dropout_add_scale_fused_inference, bias_dropout_add_scale_fused_train
from models.registry import models


import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@models.register("encoder")
class Encoder(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, num_encoder_stacks):
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
        self.k_register_latents = config.model.k_register_latents
        self.vocab_embed = EmbeddingLayer(dim,vocab_size)
        self.mask_index = vocab_size-1


        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        self.register_latent_bias = nn.Parameter(torch.zeros(self.k_register_latents, dim))
        blocks = []


        for _ in range(num_encoder_stacks):
            block = DDiTBlock(
                dim=dim,
                n_heads=config.model.n_heads,
                cond_dim=cond_dim,
                adaLN=self.adaLN,
                dropout=config.model.dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.final_normalization = nn.LayerNorm(dim)


    def _get_initialize_latents(self, batch_size, device):
        rand_latents = torch.randn(batch_size, self.k_register_latents, self.dim, device=device)
        return rand_latents + self.register_latent_bias[None, :, :]


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
                input_tokens,
                sigma,
                **kwargs):

        t_cond = None
        x = self.vocab_embed(input_tokens)
        register_latents = self._get_initialize_latents(x.shape[0], x.device)
        x = torch.cat([register_latents, x], dim=1)


        # t_cond = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
        x = self.final_normalization(x)
        # this is the encoder -- so we only return the register latents
        return x[:, :self.k_register_latents, :]

    def get_embedding(self, x):
        return self.vocab_embed(x)

    def get_mask_embed(self):
        return self.vocab_embed.embedding[self.mask_index]