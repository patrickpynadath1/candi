# Assumes that the vocab size accounts for the mask token 
from models.decoder import Decoder
from models.denoiser import Denoiser
from models.encoder import Encoder
from models.registry import models


import huggingface_hub
import omegaconf
import torch.nn as nn
import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@models.register("encoder_decoder_denoiser")
class EncoderDecoderDenoiser(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
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

        num_encoder_stacks = config.model.n_blocks // 2
        num_decoder_stacks = config.model.n_blocks - num_encoder_stacks

        self.encoder = Encoder(config, vocab_size, num_encoder_stacks)
        self.decoder = Decoder(config, vocab_size, num_decoder_stacks)
        self.denoiser = Denoiser(config, vocab_size, num_decoder_stacks)

    def encode(self, **kwargs):
        return self.encoder(**kwargs)

    def denoise(self, **kwargs):
        return self.denoiser(**kwargs, generation_mode=True)

    def decode(self, **kwargs):
        return self.decoder(**kwargs)

    def forward(self, x, sigma, return_latents=False, add_noise=False):
        latents = self.encoder(input_tokens=x, sigma=sigma.squeeze(), generation_mode=False)
        logits = self.decoder(input_latent=latents, sigma=sigma.squeeze())
        if return_latents:
            return latents, logits
        return logits