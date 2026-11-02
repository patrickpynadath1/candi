"""A terminal encoder stage that only supplies attention keys and values."""

from torch import nn

from core.models.layers import LayerNorm, rotate_half


class EncoderKeyValueReadout(nn.Module):
    """Equivalent to a DDiT encoder block's K/V, without its unused updates.

    Uses the encoder block call contract so full and incremental encoding share
    their execution loop. The hidden stream passes through unchanged; this stage
    has no query projection, attention output, MLP, or residual update.
    """

    def __init__(self, hidden_size, n_heads):
        super().__init__()
        if hidden_size % n_heads or (hidden_size // n_heads) % 2:
            raise ValueError("head dimension must be integral and even")
        self.n_heads = n_heads
        self.norm1 = LayerNorm(hidden_size)
        self.attn_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

    def forward(
        self,
        hidden,
        rotary,
        c=None,
        attention_mask=None,
        key_value_prefix=None,
        return_key_value=False,
    ):
        # A pointwise readout neither consumes context nor updates hidden states.
        del c, attention_mask, key_value_prefix
        if not return_key_value:
            return hidden
        key, value = (
            self.attn_kv(self.norm1(hidden))
            .reshape(*hidden.shape[:2], 2, self.n_heads, -1)
            .unbind(2)
        )
        cos, sin = rotary
        cos, sin = cos[:, :, 0].to(key.dtype), sin[:, :, 0].to(key.dtype)
        key = key * cos + rotate_half(key) * sin
        return hidden, (key, value)
