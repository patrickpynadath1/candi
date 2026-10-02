"""DiT layers ported locally for the noncausal MDLM architecture."""

import math
import typing
import einops
import torch
from torch import nn
from torch.nn import functional as F


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1, position_ids=None):
        if position_ids is not None:
            positions = position_ids.to(device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("bs,j->bsj", positions, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            cos = emb.cos()[:, :, None, None, :].repeat(1, 1, 3, 1, 1)
            sin = emb.sin()[:, :, None, None, :].repeat(1, 1, 3, 1, 1)
            cos[:, :, 2].fill_(1.0)
            sin[:, :, 2].fill_(0.0)
            return cos, sin
        seq_len = x.shape[seq_dim]
        if (
            seq_len != self.seq_len_cached
            or self.cos_cached is None
            or self.cos_cached.device != x.device
        ):
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def regular_attention_multi_headed(q, k, v, is_causal=False, attention_mask=None):
    # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
    # where the 3 represents Q, K, V packed in that order
    attention_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    # [batch_size, seq_len, num_heads, head_dim]
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, "b s h d -> b s (h d)")


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        adaLN,
        cond_dim=None,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c=None, attention_mask=None,
                key_value_prefix=None, return_key_value=False):

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            # self.adaLN_modulation(c): (128, 1536)
            # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
            # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
            modulation = self.adaLN_modulation(c)
            if modulation.ndim == 2:
                modulation = modulation[:, None]
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = modulation.chunk(6, dim=-1)
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = einops.rearrange(
            self.attn_qkv(x),
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.n_heads,
        )
        cos, sin = rotary_cos_sin
        # Same split-half RoPE as the legacy FlashAttention torch helper;
        # native tensor operations also permit CPU verification.
        cos, sin = cos[:, :, 0].to(qkv.dtype), sin[:, :, 0].to(qkv.dtype)
        q, k, v = qkv.unbind(dim=2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        current_key_value = (k, v)
        if key_value_prefix is not None:
            k = torch.cat([key_value_prefix[0], k], dim=1)
            v = torch.cat([key_value_prefix[1], v], dim=1)
        x = regular_attention_multi_headed(q, k, v, attention_mask=attention_mask)

        if self.adaLN:
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, gate_msa, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None,
                gate_mlp,
                x,
                self.dropout,
            )
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout
            )
        return (x, current_key_value) if return_key_value else x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            torch.nn.functional.softmax(x, dim=-1).float(),
            self.embedding.float(),
        ).to(x.dtype)


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaLN):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c, c_anchor=None):
        x = self.norm_final(x)
        if self.adaLN:
            modulation = self.adaLN_modulation(c)
            if c_anchor is not None:
                modulation = modulation - self.adaLN_modulation(c_anchor)
            if modulation.ndim == 2:
                modulation = modulation[:, None]
            shift, scale = modulation.chunk(2, dim=-1)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x
