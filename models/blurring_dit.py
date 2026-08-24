"""Bidirectional DiT with separate input and output vocabulary geometry."""

import math

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DDiTBlock, Rotary, TimestepEmbedder
from .registry import models


@models.register("blurring_dit")
class BlurringDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """Consume small blurred embeddings and classify with a larger untied table."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        hidden = int(config.model.hidden_size)
        cond_dim = int(config.model.cond_dim)
        self.input_dim = int(config.algo.input_embedding_dim)
        self.output_dim = int(config.algo.output_embedding_dim)
        self.reserve_mask_embedding = bool(
            config.algo.get("reserve_mask_embedding", False)
        )

        self.input_embedding = nn.Embedding(vocab_size, self.input_dim)
        self._initialize_input_embedding()
        self.input_projection = nn.Linear(self.input_dim, hidden)
        self.time_embedding = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden // config.model.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=hidden,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=True,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.output_projection = nn.Linear(hidden, self.output_dim, bias=False)
        self.output_embedding = nn.Parameter(torch.empty(vocab_size, self.output_dim))
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.output_embedding, std=self.output_dim**-0.5)

    def _initialize_input_embedding(self):
        """Column-aware initialization for the declared feature ordering."""
        feature_min = float(self.config.algo.get("feature_lambda_min", 1.0))
        feature_max = float(self.config.algo.get("feature_lambda_max", 1.0))
        alpha = float(self.config.algo.get("column_init_alpha", 0.0))
        if feature_min <= 0 or feature_max < feature_min or alpha < 0:
            raise ValueError("feature rates must be positive/ordered and alpha >= 0")
        # Large declared rates die first. There is deliberately no covariance
        # inversion: column 0 has the smallest rate and survives longest.
        rates = torch.logspace(
            math.log10(feature_min),
            math.log10(feature_max),
            self.input_dim,
        )
        column_scale = (rates[0] / rates).pow(alpha)
        nn.init.normal_(self.input_embedding.weight, std=self.input_dim**-0.5)
        with torch.no_grad():
            self.input_embedding.weight.mul_(column_scale[None, :])
        self.project_input_table_(iterations=64)

    @torch.no_grad()
    def project_input_table_(self, iterations=4):
        """Project data rows toward centered columns and unit row norms."""
        data = (
            self.input_embedding.weight[:-1]
            if self.reserve_mask_embedding
            else self.input_embedding.weight
        )
        for _ in range(iterations):
            data.sub_(data.mean(dim=0, keepdim=True))
            data.div_(data.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        data.sub_(data.mean(dim=0, keepdim=True))
        # The final centering perturbation is O(1/V); normalize-on-use below
        # makes the row-norm constraint exact in every forward process.
        if self.reserve_mask_embedding:
            mask = self.input_embedding.weight[-1]
            mask.div_(mask.norm().clamp_min(1e-12))

    def input_column_norms(self):
        return self.data_input_table().norm(dim=0)

    def normalized_input_table(self):
        """Normalize on use so every forward process sees unit-length rows."""
        return F.normalize(self.input_embedding.weight, dim=-1)

    def data_input_table(self):
        table = self.normalized_input_table()
        return table[:-1] if self.reserve_mask_embedding else table

    def get_embedding(self, values, *, detach=False):
        """Embed integer tokens or blend the small table with token probabilities."""
        table = self.data_input_table()
        if detach:
            table = table.detach()
        if not values.dtype.is_floating_point:
            return F.embedding(values, table)
        if values.shape[-1] != table.shape[0]:
            raise ValueError("probability blends must end in the vocabulary dimension")
        return values.to(table.dtype) @ table

    def forward(self, xt, time, **_kwargs):
        if xt.ndim != 3 or xt.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [batch, length, {self.input_dim}] blurred embeddings"
            )
        condition = F.silu(self.time_embedding(time))
        hidden = self.input_projection(xt)
        rotary = self.rotary_emb(hidden)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            for block in self.blocks:
                hidden = block(hidden, rotary, c=condition)
            features = self.output_projection(hidden)
            logits = features @ self.output_embedding.t() + self.output_bias
        return logits[:, :, :-1] if self.reserve_mask_embedding else logits
