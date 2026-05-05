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

@models.register("embedding_wrapper")
class EmbeddingWrapper(nn.Module, huggingface_hub.PyTorchModelHubMixin):
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
        # self.linear = nn.Linear(dim, dim)
        self.embedding = nn.Embedding(config.algo.num_codes, dim)

        # self.vq = instantiate(config.algo.vq)

    def forward(self, z):
        # logits 
        return self.embedding(z)
        