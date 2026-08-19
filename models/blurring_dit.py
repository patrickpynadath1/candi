"""Bidirectional DiT with separate input and output vocabulary geometry."""

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

        self.input_embedding = nn.Embedding(vocab_size, self.input_dim)
        nn.init.normal_(self.input_embedding.weight, std=self.input_dim ** -0.5)
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
        self.output_embedding = nn.Parameter(
            torch.empty(vocab_size, self.output_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.output_embedding, std=self.output_dim ** -0.5)

    def normalized_input_table(self):
        """Normalize on use so every forward process sees unit-length rows."""
        return F.normalize(self.input_embedding.weight, dim=-1)

    def get_embedding(self, values, *, detach=False):
        """Embed integer tokens or blend the small table with token probabilities."""
        table = self.normalized_input_table()
        if detach:
            table = table.detach()
        if not values.dtype.is_floating_point:
            return F.embedding(values, table)
        if values.shape[-1] != self.vocab_size:
            raise ValueError(
                "probability blends must end in the vocabulary dimension"
            )
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
        return logits
