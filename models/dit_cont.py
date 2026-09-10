from .common import (
    EmbeddingLayerMixture_norm,
    TimestepEmbedder,
    Rotary,
    DDiTBlock,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    Preconditioning
)
from .registry import models
import math
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
@models.register("cont_dit")
class ContDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, mixed_coeff: float=.5, use_precond=True):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = True
        self.config = config
        self.vocab_size = vocab_size
        self.reserve_mask_embedding = getattr(
            config.algo, "reserve_mask_embedding", True
        )
        dim = int(config.model.hidden_size)
        self.dim = dim
        self.input_dim = int(getattr(config.algo, "input_embedding_dim", dim))
        self.output_dim = int(getattr(config.algo, "output_embedding_dim", dim))
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayerMixture_norm(
            self.input_dim,
            vocab_size,
            reserve_mask_embedding=self.reserve_mask_embedding,
        )
        self._initialize_input_embedding()
        self.input_projection = nn.Linear(self.input_dim, dim)
        self.mask_index = vocab_size - 1 if self.reserve_mask_embedding else None
        self.use_precond = use_precond


        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        self.mixed_coeff = mixed_coeff
        self.precond = Preconditioning(self.vocab_embed, self.mixed_coeff)



        blocks = []
        for _ in range(config.model.n_blocks):
            block = DDiTBlock(
                dim=dim,
                n_heads=config.model.n_heads,
                cond_dim=cond_dim,
                adaLN=self.adaLN,
                dropout=config.model.dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # Match BlurringDIT's untied, factorized output geometry. In
        # particular, do not place a second AdaLN block or a zero-initialized
        # classifier after the transformer.
        self.output_projection = nn.Linear(dim, self.output_dim, bias=False)
        self.output_embedding = nn.Parameter(
            torch.empty(vocab_size, self.output_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.output_embedding, std=self.output_dim**-0.5)

    def _algo_value(self, name, default):
        if hasattr(self.config.algo, "get"):
            return self.config.algo.get(name, default)
        return getattr(self.config.algo, name, default)

    def _initialize_input_embedding(self):
        """Use BlurringDIT's centered, unit-row input-table geometry."""
        feature_min = float(self._algo_value("feature_lambda_min", 1.0))
        feature_max = float(self._algo_value("feature_lambda_max", 1.0))
        alpha = float(self._algo_value("column_init_alpha", 0.0))
        if feature_min <= 0 or feature_max < feature_min or alpha < 0:
            raise ValueError("feature rates must be positive/ordered and alpha >= 0")
        rates = torch.logspace(
            math.log10(feature_min),
            math.log10(feature_max),
            self.input_dim,
        )
        column_scale = (rates[0] / rates).pow(alpha)
        nn.init.normal_(self.vocab_embed.embedding, std=self.input_dim**-0.5)
        with torch.no_grad():
            self.vocab_embed.embedding.mul_(column_scale[None, :])
        self.project_input_table_(iterations=64)

    @torch.no_grad()
    def project_input_table_(self, iterations=4):
        data = (
            self.vocab_embed.embedding[:-1]
            if self.reserve_mask_embedding
            else self.vocab_embed.embedding
        )
        for _ in range(iterations):
            data.sub_(data.mean(dim=0, keepdim=True))
            data.div_(data.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        data.sub_(data.mean(dim=0, keepdim=True))
        if self.reserve_mask_embedding:
            mask = self.vocab_embed.embedding[-1]
            mask.div_(mask.norm().clamp_min(1e-12))

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
        
    # clean mask: indicates which positions are clean with 1, which are corrupted with 0
    # def forward(self, x, discrete_noise):
    def forward(self, xt, 
                discrete_noise, 
                reveal_mask, 
                continuous_noise=None, 
                embedding=None, 
                ret_embedding=False, 
                is_embed=False, 
                precond=True,
        **kwargs):
        x = xt 
        # The public time input is a corruption probability p. Preserve the
        # native -log(1-p) condition while keeping the exact p=1 sampling
        # endpoint numerically finite.
        time_eps = float(getattr(self.config.algo, "eps", 1e-6))
        probability = discrete_noise.float().clamp(
            min=0.0, max=1.0 - max(time_eps, torch.finfo(torch.float32).eps)
        )
        sigma = -torch.log1p(-probability)
        if not is_embed:
            x = self.vocab_embed(x)
        if precond:
            x = self.precond(x, continuous_noise=continuous_noise, embeds=embedding, reveal_mask=reveal_mask)

        t_cond = F.silu(self.sigma_map(sigma))
        hidden = self.input_projection(x)
        rotary_cos_sin = self.rotary_emb(hidden)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            for i in range(len(self.blocks)):
                hidden = self.blocks[i](hidden, rotary_cos_sin, c=t_cond)
            features = self.output_projection(hidden)
            logits = features @ self.output_embedding.t() + self.output_bias

        data_logits = (
            logits[:, :, :-1] if self.reserve_mask_embedding else logits
        )
        if ret_embedding:
            return data_logits, hidden
        return data_logits

    def get_embedding(self, x):
        return self.vocab_embed(x)

    def get_mask_embed(self):
        if not self.reserve_mask_embedding:
            raise RuntimeError("This ContDIT has no reserved mask embedding.")
        return self.vocab_embed.embedding[self.mask_index]
