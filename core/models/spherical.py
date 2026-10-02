import torch
import torch.nn as nn

from .prediction import make_conditioning
from .text_dit import TextDiT
from .components import LinearStateInput, TemperatureOutput
from .layers import DDiTBlock, DDiTFinalLayer, Rotary, TimestepEmbedder


class SphereDiT(TextDiT):
    def __init__(
        self,
        vocab_size,
        state_dim=512,
        hidden_size=512,
        cond_dim=128,
        n_heads=8,
        n_blocks=8,
        dropout=0.1,
        learn_temperature_scaling=False,
        self_conditioning=False,
        self_conditioning_source="posterior_mean",
        input_adapter=None,
        time_adapter=None,
        output_adapter=None,
        block_execution=None,
    ):
        super().__init__()
        if hidden_size % n_heads or (hidden_size // n_heads) % 2:
            raise ValueError("hidden_size must give an even integral head dimension")
        dim = hidden_size
        self.embed_dim = self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_size
        self.input_adapter = LinearStateInput()
        self.input_projection = nn.Linear(self.embed_dim, self.hidden_dim)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim,
                    n_heads,
                    adaLN=True,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.learn_temperature = learn_temperature_scaling
        if self.learn_temperature:
            self.output_adapter = TemperatureOutput()
        self.output_layer = DDiTFinalLayer(
            dim, vocab_size + int(self.learn_temperature), cond_dim, True
        )

        self.conditioning_adapter = (
            make_conditioning(self_conditioning_source, state_dim, hidden_size)
            if self_conditioning
            else None
        )

        self.configure_components(
            input_adapter=input_adapter, time_adapter=time_adapter,
            output_adapter=output_adapter, block_execution=block_execution,
        )

    def forward(self, x, sigma, *, self_condition=None, return_features=False):
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected [batch, length, {self.embed_dim}] spherical embeddings"
            )
        with torch.autocast(x.device.type, enabled=False):
            cond = self.time_adapter(sigma.float(), self.sigma_map)
            hidden = self.input_adapter(x, self.input_projection)
            if self.conditioning_adapter is not None:
                hidden = self.conditioning_adapter(hidden, self_condition)
            elif self_condition is not None:
                raise ValueError("model does not enable self-conditioning")
        return self.forward_hidden(hidden, cond, return_features=return_features)
