import math
import pdb
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
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


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
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

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
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


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
    with torch.cuda.amp.autocast(enabled=False):
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        q, k, v = qkv.chunk(3, dim=2)
        q = flash_attn.layers.rotary.apply_rotary_emb_torch(q.squeeze(dim=2), cos, sin)
        k = flash_attn.layers.rotary.apply_rotary_emb_torch(k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
    # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
    # where the 3 represents Q, K, V packed in that order
    attention_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    # [batch_size, seq_len, num_heads, head_dim]
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, "b s h d -> b s (h d)")


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
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


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlockCausal(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

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

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, **kwargs):
        del kwargs
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device=qkv.device,
        )
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=True
        )

        x = einops.rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, adaLN, cond_dim=None, mlp_ratio=4, dropout=0.1):
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

    def forward(self, x, rotary_cos_sin, c=None):

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            # self.adaLN_modulation(c): (128, 1536)
            # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
            # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = einops.rearrange(
            self.attn_qkv(x),
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.n_heads,
        )
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

        x = regular_attention_multi_headed(q, k, v)

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
        return x


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


    
# instead of doing vocab lookup, 
# does weighted average
class EmbeddingLayerMixture_norm(nn.Module):

    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):

        # Just for now 
        normalized_embeddings = F.normalize(self.embedding, p=2, dim=-1)
        # normalized_embeddings = normalized_embeddings * math.sqrt(self.dim)
        
        # normalized_embeddings = self.embedding[:-1]

        if x.ndim == 2:
            return normalized_embeddings[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            x.float(),
            normalized_embeddings.float(),
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
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


class Preconditioning(nn.Module):
    def __init__(self, embed_table, mixed_coeff, masked_index=-1):
        super().__init__()
        self.embed_table = embed_table
        self.mixed_coeff = mixed_coeff
        self.masked_index = masked_index

    def forward(self, x, continuous_noise, reveal_mask=None, embeds=None):
        c_in = 1 / (1 + continuous_noise ** 2) ** .5
        if reveal_mask is not None:
            special = self.embed_table.embedding[self.masked_index].view(1, 1, -1).expand_as(x)
            mask = reveal_mask.unsqueeze(-1).float()
            coeffs = torch.ones(mask.size(0), device=mask.device)[:, None, None] * self.mixed_coeff
            if embeds is not None: 
                x = x * mask + (1 - mask) * (coeffs * special / special.norm(dim=-1, keepdim=True)+ (1 - coeffs) * embeds * c_in[:, None, None]) 
            else: 
                x = x * mask + (1 - mask) * (coeffs * special / special.norm(dim=-1, keepdim=True)+ (1 - coeffs) * x * c_in[:, None, None])
            return x
        return x * c_in[:, None, None]



class CrossAttentionEncoderBlock(nn.Module):
    """Compresses normal sequence into a smaller latent sequence.
    
    Learnable latent queries cross-attend to the input sequence.
    """
    def __init__(self, dim, n_heads, n_latents, adaLN=False, cond_dim=None, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN

        # Learnable latent queries: (n_latents, dim)
        self.latent_queries = nn.Parameter(torch.randn(n_latents, dim) * 0.02)

        # Cross attention projections
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # MLP
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        if self.adaLN:
            # 6 modulation params: shift/scale/gate for cross-attn and mlp
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def _cross_attention(self, q, k, v):
        """q: (b, s_q, h, d), k: (b, s_kv, h, d), v: (b, s_kv, h, d)"""
        scale = q.shape[-1] ** -0.5
        # (b, h, s_q, s_kv)
        attn = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        # (b, s_q, h, d)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        return einops.rearrange(out, "b s h d -> b s (h d)")

    def forward(self, x, c=None):
        """
        x: (b, s, dim) - normal sequence (keys/values)
        c: (b, cond_dim) - conditioning (optional)
        Returns: (b, n_latents, dim) - compressed latent
        """
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        b = x.shape[0]

        # Expand learnable queries to batch: (b, n_latents, dim)
        latent = self.latent_queries.unsqueeze(0).expand(b, -1, -1)

        latent_skip = latent

        # Norms
        q_in = self.norm_q(latent)
        kv_in = self.norm_kv(x)

        if self.adaLN:
            (shift_ca, scale_ca, gate_ca,
             shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            q_in = modulate_fused(q_in, shift_ca, scale_ca)

        # Project q from latent, k/v from input
        q = einops.rearrange(self.attn_q(q_in), "b s (h d) -> b s h d", h=self.n_heads)
        kv = einops.rearrange(self.attn_kv(kv_in), "b s (two h d) -> b s two h d", two=2, h=self.n_heads)
        k, v = kv[:, :, 0], kv[:, :, 1]

        attn_out = self._cross_attention(q, k, v)

        if self.adaLN:
            latent = bias_dropout_scale_fn(
                self.attn_out(attn_out), None, gate_ca, latent_skip, self.dropout
            )
            latent = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(latent), shift_mlp, scale_mlp)),
                None, gate_mlp, latent, self.dropout
            )
        else:
            scale = torch.ones(1, device=latent.device, dtype=latent.dtype)
            latent = bias_dropout_scale_fn(
                self.attn_out(attn_out), None, scale, latent_skip, self.dropout
            )
            latent = bias_dropout_scale_fn(
                self.mlp(self.norm2(latent)), None, scale, latent, self.dropout
            )

        return latent


class CrossAttentionDecoderBlock(nn.Module):
    """Expands latent sequence back to normal sequence length.
    
    Learnable output queries cross-attend to the latent sequence.
    """
    def __init__(self, dim, n_heads, n_outputs, adaLN=False, cond_dim=None, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN

        # Learnable output queries: (n_outputs, dim) — n_outputs matches original seq len
        self.output_queries = nn.Parameter(torch.randn(n_outputs, dim) * 0.02)

        # Cross attention projections
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.query_positions = nn.Embedding(n_outputs, dim)

        # MLP
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

    def _cross_attention(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        attn = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        return einops.rearrange(out, "b s h d -> b s (h d)")

    def forward(self, latent, c=None):
        """
        latent: (b, n_latents, dim) - compressed latent (keys/values)
        c: (b, cond_dim) - conditioning (optional)
        Returns: (b, n_outputs, dim) - expanded output
        """
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        b = latent.shape[0]

        # Expand learnable output queries: (b, n_outputs, dim)
        out = self.output_queries.unsqueeze(0).expand(b, -1, -1)

        out_skip = out

        q_in = self.norm_q(out)
        kv_in = self.norm_kv(latent)

        if self.adaLN:
            (shift_ca, scale_ca, gate_ca,
             shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            q_in = modulate_fused(q_in, shift_ca, scale_ca)

        q = einops.rearrange(self.attn_q(q_in), "b s (h d) -> b s h d", h=self.n_heads)
        kv = einops.rearrange(self.attn_kv(kv_in), "b s (two h d) -> b s two h d", two=2, h=self.n_heads)
        k, v = kv[:, :, 0], kv[:, :, 1]

        attn_out = self._cross_attention(q, k, v)

        if self.adaLN:
            out = bias_dropout_scale_fn(
                self.attn_out(attn_out), None, gate_ca, out_skip, self.dropout
            )
            out = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(out), shift_mlp, scale_mlp)),
                None, gate_mlp, out, self.dropout
            )
        else:
            scale = torch.ones(1, device=out.device, dtype=out.dtype)
            out = bias_dropout_scale_fn(
                self.attn_out(attn_out), None, scale, out_skip, self.dropout
            )
            out = bias_dropout_scale_fn(
                self.mlp(self.norm2(out)), None, scale, out, self.dropout
            )

        return out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
