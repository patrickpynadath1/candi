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
        sublayer_factory=None,
        mlp_hidden_size=None,
    ):
        super().__init__()
        if sublayer_factory is not None and mlp_hidden_size is not None:
            raise ValueError("mlp_hidden_size cannot be combined with sublayer_factory")
        self.n_heads = n_heads
        self.adaLN = adaLN

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        from .sublayers import PositionwiseMLP

        self.mlp = (
            PositionwiseMLP(dim, mlp_ratio, hidden_size=mlp_hidden_size)
            if sublayer_factory is None
            else sublayer_factory(dim=dim, n_heads=n_heads)
        )
        self.positionwise_sublayer = isinstance(self.mlp, PositionwiseMLP)
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

    def project_context_key_value(self, context, rotary_cos_sin):
        """Project an external hidden context with this layer's K/V weights."""
        context = self.norm1(context)
        _, key_weight, value_weight = self.attn_qkv.weight.chunk(3, dim=0)
        shape = (
            *context.shape[:2],
            self.n_heads,
            self.attn_qkv.in_features // self.n_heads,
        )
        key = F.linear(context, key_weight).reshape(shape)
        value = F.linear(context, value_weight).reshape(shape)
        cos, sin = rotary_cos_sin
        cos, sin = cos[:, :, 0].to(key.dtype), sin[:, :, 0].to(key.dtype)
        return key * cos + rotate_half(key) * sin, value

    def forward(
        self,
        x,
        rotary_cos_sin,
        c=None,
        attention_mask=None,
        key_value_prefix=None,
        return_key_value=False,
        context_attention=None,
    ):
        output, key_value, _ = self._forward(
            x,
            rotary_cos_sin,
            c,
            attention_mask,
            key_value_prefix=key_value_prefix,
            return_key_value=return_key_value,
            context_attention=context_attention,
        )
        return (output, key_value) if return_key_value else output

    def forward_streams(self, streams, rotary_cos_sin, c=None, attention_mask=None):
        """Consume Q/K/V/residual sources; also expose the residual contribution.

        Depth aggregation changes these inputs, not attention or the MLP.
        Returning the contribution directly avoids subtracting large residuals
        when a depth policy stores branch updates rather than whole states.
        """
        query, key, value, residual = streams.unbind(dim=2)
        output, _, contribution = self._forward(
            query,
            rotary_cos_sin,
            c,
            attention_mask,
            key_input=key,
            value_input=value,
            residual=residual,
        )
        return output, contribution

    def _forward(
        self,
        x,
        rotary_cos_sin,
        c,
        attention_mask,
        *,
        key_input=None,
        value_input=None,
        residual=None,
        key_value_prefix=None,
        return_key_value=False,
        context_attention=None,
    ):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        if not self.positionwise_sublayer and (
            key_value_prefix is not None or return_key_value
        ):
            raise ValueError("second attention sublayers do not yet support KV caching")
        x_skip = x if residual is None else residual
        if self.adaLN:
            modulation = self.adaLN_modulation(c)
            if modulation.ndim == 2:
                modulation = modulation[:, None]
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
                modulation.chunk(6, dim=-1)
            )
        else:
            gate_msa = gate_mlp = torch.ones(1, device=x.device, dtype=x.dtype)

        def normalize_attention(source):
            source = self.norm1(source)
            return (
                modulate_fused(source, shift_msa, scale_msa) if self.adaLN else source
            )

        if key_input is None:
            q, k, v = (
                self.attn_qkv(normalize_attention(x))
                .reshape(*x.shape[:2], 3, self.n_heads, -1)
                .unbind(2)
            )
            cos, sin = rotary_cos_sin
            cos, sin = cos[:, :, 0].to(q.dtype), sin[:, :, 0].to(q.dtype)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        else:
            # Same projections with distinct depth-mixed source streams.
            weights = self.attn_qkv.weight.chunk(3, dim=0)
            q, k, v = (
                F.linear(normalize_attention(source), weight).reshape(
                    *source.shape[:2], self.n_heads, -1
                )
                for source, weight in zip((x, key_input, value_input), weights)
            )
            cos, sin = rotary_cos_sin
            cos, sin = cos[:, :, 0].to(q.dtype), sin[:, :, 0].to(q.dtype)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        current_key_value = (k, v)
        if key_value_prefix is not None and context_attention is not None:
            attended = context_attention(q, k, v, key_value_prefix, attention_mask)
        else:
            if key_value_prefix is not None:
                k = torch.cat([key_value_prefix[0], k], dim=1)
                v = torch.cat([key_value_prefix[1], v], dim=1)
            attended = regular_attention_multi_headed(
                q, k, v, attention_mask=attention_mask
            )
        attention_update = bias_dropout_scale_fn(
            self.attn_out(attended), None, gate_msa, None, self.dropout
        )
        hidden = x_skip + attention_update
        mlp_input = self.norm2(hidden)
        if self.adaLN:
            mlp_input = modulate_fused(mlp_input, shift_mlp, scale_mlp)
        mlp_update = bias_dropout_scale_fn(
            self.mlp(
                mlp_input, rotary_cos_sin=rotary_cos_sin, attention_mask=attention_mask
            ),
            None,
            gate_mlp,
            None,
            self.dropout,
        )
        return hidden + mlp_update, current_key_value, attention_update + mlp_update


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

    def forward(self, x, c):
        x = self.norm_final(x)
        if self.adaLN:
            modulation = self.adaLN_modulation(c)
            if modulation.ndim == 2:
                modulation = modulation[:, None]
            shift, scale = modulation.chunk(2, dim=-1)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x
