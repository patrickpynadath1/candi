from .common import (
    EmbeddingLayerMixture_norm,
    TimestepEmbedder,
    Rotary,
    DDiTBlock,
    DDiTFinalLayer,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
)
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



# Assumes that the vocab size accounts for the mask token 
class ContDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, mixed_coeff: float=.5, use_precond=True):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = True
        self.config = config
        self.vocab_size = vocab_size
        dim = config.hidden_size
        self.dim=dim
        cond_dim = config.cond_dim
        self.vocab_embed = EmbeddingLayerMixture_norm(dim,vocab_size)
        self.mask_index = vocab_size
        self.use_precond = use_precond


        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.n_heads)
        self.mixed_coeff = mixed_coeff



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
        
    # clean mask: indicates which positions are clean with 1, which are corrupted with 0
    # def forward(self, x, discrete_noise):
    def forward(self, xt, discrete_noise, reveal_mask, continuous_noise, embedding=None, ret_embedding=False, is_embed=False):
        x = xt 
        sigma = -1 * torch.log(1 - discrete_noise)
        c_in = 1 / (1 + continuous_noise ** 2) ** .5
        if is_embed == False:
            x = self.vocab_embed(x)

        special = self.vocab_embed.embedding[-1].view(1, 1, -1).expand_as(x)
        mask = reveal_mask.unsqueeze(-1).float()
        coeffs = torch.ones(mask.size(0), device=mask.device)[:, None, None] * self.mixed_coeff 


        if embedding is not None: 
            x = x * mask + (1 - mask) * (coeffs * special / special.norm(dim=-1, keepdim=True)+ (1 - coeffs) * embedding * c_in[:, None, None]) 
        else:
            x = x * mask + (1 - mask) * (coeffs * special / special.norm(dim=-1, keepdim=True)+ (1 - coeffs) * x * c_in[:, None, None])

        # x = c_in[:, None, None] * x
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            logits = self.output_layer(x, c=t_cond)

        if ret_embedding:
            return logits, x 
        return logits
    
    def get_embedding(self, x): 
        return self.vocab_embed(x)
    