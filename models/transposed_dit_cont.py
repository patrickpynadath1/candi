"""Continuous DiT whose attention axis is the embedding-coordinate axis."""

from .common import (
    EmbeddingLayerMixture_norm,
    TimestepEmbedder,
    Rotary,
    DDiTBlock,
    DDiTFinalLayer,
    Preconditioning,
)
from .registry import models

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityRotary(nn.Module):
    """Return identity rotary factors while retaining the DDiT block API."""

    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x):
        shape = (1, x.shape[1], 3, 1, self.head_dim)
        return (
            torch.ones(shape, device=x.device, dtype=x.dtype),
            torch.zeros(shape, device=x.device, dtype=x.dtype),
        )


@models.register("transposed_cont_dit")
class TransposedContDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """Attend over embedding coordinates instead of sequence positions.

    The input adapter maps ``[batch, length, embedding_dim]`` to
    ``[batch, embedding_dim, transformer_dim]``. The output adapter reverses
    that operation before the usual per-position vocabulary projection.
    """

    def __init__(self, config, vocab_size: int, mixed_coeff: float = 0.5, use_precond=True):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = True
        self.config = config
        self.vocab_size = vocab_size
        self.length = config.model.length
        self.embedding_dim = config.model.hidden_size
        self.dim = self.embedding_dim
        transformer_dim = getattr(
            config.model, "transpose_hidden_size", self.embedding_dim
        )
        self.transformer_dim = transformer_dim
        n_heads = config.model.n_heads
        if transformer_dim % n_heads:
            raise ValueError(
                "model.transpose_hidden_size must be divisible by model.n_heads"
            )

        self.reserve_mask_embedding = getattr(
            config.algo, "reserve_mask_embedding", True
        )
        self.vocab_embed = EmbeddingLayerMixture_norm(
            self.embedding_dim,
            vocab_size,
            reserve_mask_embedding=self.reserve_mask_embedding,
        )
        self.mask_index = vocab_size - 1 if self.reserve_mask_embedding else None
        self.use_precond = use_precond
        self.mixed_coeff = mixed_coeff
        self.precond = Preconditioning(self.vocab_embed, mixed_coeff)

        cond_dim = config.model.cond_dim
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.sequence_up = nn.Linear(self.length, transformer_dim)
        self.sequence_down = nn.Linear(transformer_dim, self.length)

        position_encoding = getattr(
            config.model, "coordinate_position_encoding", "none"
        )
        if position_encoding not in {"none", "rope", "learned"}:
            raise ValueError(
                "model.coordinate_position_encoding must be none, rope, or learned"
            )
        self.coordinate_position_encoding = position_encoding
        if position_encoding == "rope":
            self.rotary_emb = Rotary(transformer_dim // n_heads)
        else:
            self.rotary_emb = IdentityRotary(transformer_dim // n_heads)
        if position_encoding == "learned":
            self.coordinate_embedding = nn.Parameter(
                torch.empty(self.embedding_dim, transformer_dim)
            )
            nn.init.normal_(self.coordinate_embedding, std=0.02)
        else:
            self.register_parameter("coordinate_embedding", None)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=transformer_dim,
                    n_heads=n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.output_layer = DDiTFinalLayer(
            hidden_size=self.embedding_dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )

    def forward(
        self,
        xt,
        discrete_noise,
        reveal_mask,
        continuous_noise=None,
        embedding=None,
        ret_embedding=False,
        is_embed=False,
        precond=True,
        **kwargs,
    ):
        del kwargs
        x = xt if is_embed else self.vocab_embed(xt)
        if x.ndim != 3:
            raise ValueError("expected an input with shape [batch, length, embedding_dim]")
        if x.shape[1] != self.length:
            raise ValueError(
                f"transposed_cont_dit requires sequence length {self.length}, "
                f"but received {x.shape[1]}"
            )
        if x.shape[2] != self.embedding_dim:
            raise ValueError(
                f"expected embedding width {self.embedding_dim}, but received {x.shape[2]}"
            )
        if precond:
            x = self.precond(
                x,
                continuous_noise=continuous_noise,
                embeds=embedding,
                reveal_mask=reveal_mask,
            )

        condition = F.silu(self.sigma_map(-torch.log1p(-discrete_noise)))
        hidden = self.sequence_up(x.transpose(1, 2))
        if self.coordinate_embedding is not None:
            hidden = hidden + self.coordinate_embedding.unsqueeze(0)
        rotary_cos_sin = self.rotary_emb(hidden)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            for block in self.blocks:
                hidden = block(hidden, rotary_cos_sin, c=condition)
            reconstructed = self.sequence_down(hidden).transpose(1, 2)
            logits = self.output_layer(reconstructed, c=condition)

        data_logits = logits[:, :, :-1] if self.reserve_mask_embedding else logits
        if ret_embedding:
            return data_logits, reconstructed
        return data_logits

    def get_embedding(self, x):
        return self.vocab_embed(x)

    def get_mask_embed(self):
        if not self.reserve_mask_embedding:
            raise RuntimeError("This TransposedContDIT has no reserved mask embedding.")
        return self.vocab_embed.embedding[self.mask_index]
