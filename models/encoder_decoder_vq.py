from .decoder import Decoder
from .encoder import Encoder
from .registry import models
from .vq import VQ_GaussResid, VQ_MixtureResidual, VQ_STE
from vector_quantize_pytorch import VectorQuantize


import huggingface_hub
import omegaconf
from hydra.utils import instantiate
import torch.nn as nn
# Flags required to enable jit fusion kernels
import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@models.register("encoder_decoder_vq")
class EncoderDecoder(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
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
        self.vq_mode = config.algo.vq_mode
        num_encoder_stacks = config.model.n_blocks // 2
        num_decoder_stacks = config.model.n_blocks - num_encoder_stacks
        # if self.vq_mode == "gauss_resid":
        #     self.vq = VQ_GaussResid(config.model.num_codes, 
        #                             config.model.hidden_size, 
        #                             config.model.residual_compression_rate)
        # elif self.vq_mode == "mixture_resid":
        #     self.vq = VQ_MixtureResidual(config.model.num_codes, 
        #                                  config.model.hidden_size, 
        #                                  config.model.residual_compression_rate)
        # elif self.vq_mode == "ste":
        #     self.vq = VQ_STE(config.model.num_codes, 
        #                      config.model.hidden_size)
        # elif self.vq_mode == "vq":
        # self.vq = instantiate(config.algo.vq)

            

        # else: 
        #     self.vq = None
        self.encoder = Encoder(config, vocab_size, num_encoder_stacks)
        self.decoder = Decoder(config, vocab_size, num_decoder_stacks)

    def encode(self, **kwargs):
        encoder_outputs =self.encoder(**kwargs)
        return encoder_outputs

    def decode(self, input_latent, **kwargs):
        return self.decoder(input_latent, **kwargs)

    def forward(self, x, sigma, return_latents=False):
        latents = self.encoder(input_tokens=x, sigma=sigma.squeeze(), generation_mode=False)
        logits = self.decoder(input_latent=latents, sigma=sigma.squeeze())
        if return_latents:
            return latents, logits
        return logits