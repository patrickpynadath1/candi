"""Shared-transformer clean/noisy stream model wrapper."""

from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch import nn

from core.models.continuous_dit import ContinuousDiT
from core.models.dit import DiT
from core.models.hybrid import HybridModel
from core.models.prediction import ModelPrediction
from .layout import make_layout


class BlockDiffusionModel(nn.Module):
    def __init__(self, model, block_size, clean_encoder=None):
        super().__init__()
        if not isinstance(model, (DiT, ContinuousDiT, HybridModel)):
            raise TypeError(
                "block diffusion requires a model with fused-stream hidden interfaces"
            )
        self.model = model
        self.block_size = block_size
        self.clean_encoder = clean_encoder
        self._clean = ContextVar("block_diffusion_clean", default=None)
        self.last_clean_logits = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @contextmanager
    def clean_context(self, tokens, *, offset=0, cache=None):
        token = self._clean.set((tokens, offset, cache))
        try:
            yield
        finally:
            self._clean.reset(token)

    @contextmanager
    def select_context(self, selected):
        current = self._clean.get()
        if current is None:
            yield
            return
        token = self._clean.set((current[0][selected], current[1], None))
        try:
            yield
        finally:
            self._clean.reset(token)

    def _split_output(self, output, length):
        clean_hidden, noisy_hidden = output.hidden[:, :length], output.hidden[:, length:]
        clean_logits, noisy_logits = output.logits[:, :length], output.logits[:, length:]
        self.last_clean_logits = clean_logits
        return ModelPrediction(noisy_logits, noisy_hidden, clean_logits, clean_hidden)

    def forward(self, *inputs, self_condition=None, return_features=False):
        current = self._clean.get()
        if current is None:
            return self.model(
                *inputs, self_condition=self_condition, return_features=return_features
            )
        clean, offset, cache = current
        noisy = inputs[0][0] if isinstance(inputs[0], (tuple, list)) else inputs[0]
        length = noisy.shape[1]
        if len(clean) != len(noisy):
            raise ValueError("clean block context must match model batch")
        clean_length = clean.shape[1]
        layout = make_layout(
            length, self.block_size, noisy.device,
            batch_size=len(inputs[0]), offset=offset, clean_length=clean_length
        )
        if cache is not None and not self.training:
            return self._cached_forward(
                inputs, clean, offset, cache, layout, self_condition, return_features
            )
        clean_hidden, clean_condition, noisy_hidden, noisy_condition = self._stream_inputs(
            inputs, clean, self_condition
        )
        condition = None
        if clean_condition is not None:
            condition = torch.cat(
                [clean_condition[:, None].expand(-1, clean_length, -1),
                 noisy_condition[:, None].expand(-1, length, -1)], 1
            )
        output = self.model.forward_hidden(
            torch.cat([clean_hidden, noisy_hidden], 1), condition,
            return_features=True, attention_mask=layout.attention_mask,
            position_ids=layout.position_ids,
        )
        clean_logits, noisy_logits = output.logits[:, :clean_length], output.logits[:, clean_length:]
        clean_hidden, noisy_hidden = output.hidden[:, :clean_length], output.hidden[:, clean_length:]
        self.last_clean_logits = clean_logits
        result = ModelPrediction(noisy_logits, noisy_hidden, clean_logits, clean_hidden)
        return result if return_features else result.logits

    def _stream_inputs(self, inputs, clean, self_condition):
        if isinstance(self.model, HybridModel):
            states, points, global_time = inputs
            noisy_hidden, noisy_condition = self.model.prepare_fused(
                states, points, global_time, self_condition=self_condition
            )
            probabilities = torch.nn.functional.one_hot(
                clean,
                self.model.output_vocab_size
                or self.model.output_layer.linear.out_features,
            ).float()
            clean_hidden, clean_condition = self.model.clean_state_hidden(probabilities)
        elif isinstance(self.model, ContinuousDiT):
            state, time, sigma, alpha = inputs
            noisy_hidden, noisy_condition = self.model.prepare_hidden(
                state, time, sigma, alpha, self_condition=self_condition
            )
            clean_state = self.clean_encoder(clean) if self.clean_encoder else clean
            clean_hidden, clean_condition = self.model.clean_state_hidden(clean_state)
        else:
            tokens, conditioning = inputs
            with torch.autocast(tokens.device.type, enabled=False):
                noisy_emb = (
                    self.model.embed_state(tokens)
                    if hasattr(self.model, "embed_state") and tokens.ndim == 3
                    else self.model.vocab_embed(tokens)
                )
                clean_hidden = self.model.vocab_embed(clean)
            noisy_hidden, noisy_condition = self.model.prepare_hidden(
                noisy_emb, conditioning, self_condition=self_condition
            )
            clean_condition = (
                torch.nn.functional.silu(
                    self.model.sigma_map(torch.zeros(len(clean), device=clean.device))
                ) if self.model.adaLN else None
            )
        return clean_hidden, clean_condition, noisy_hidden, noisy_condition

    def _cached_forward(self, inputs, clean, offset, cache, layout,
                        self_condition, return_features):
        clean_hidden, clean_condition, noisy_hidden, noisy_condition = self._stream_inputs(
            inputs, clean, self_condition
        )
        clean_length = clean.shape[1]
        if "layers" not in cache:
            layers = []
            if clean_length:
                clean_ids = layout.position_ids[:, :clean_length]
                with torch.autocast(clean.device.type, enabled=False):
                    clean_rotary = self.model.rotary_emb(
                        clean_hidden, position_ids=clean_ids
                    )
                causal = layout.attention_mask[:, :, :clean_length, :clean_length]
                with torch.autocast(
                    clean.device.type,
                    enabled=torch.is_autocast_enabled(clean.device.type),
                    cache_enabled=False,
                ):
                    for block in self.model.blocks:
                        clean_hidden, key_value = block(
                            clean_hidden, clean_rotary, clean_condition, causal,
                            return_key_value=True
                        )
                        layers.append(key_value)
            cache["layers"] = layers
            cache["clean_hidden"] = clean_hidden
        with torch.autocast(noisy_hidden.device.type, enabled=False):
            noisy_rotary = self.model.rotary_emb(
                noisy_hidden, position_ids=layout.position_ids[:, clean_length:]
            )
        noisy_mask = layout.attention_mask[:, :, clean_length:, :]
        with torch.autocast(
            noisy_hidden.device.type,
            enabled=torch.is_autocast_enabled(noisy_hidden.device.type),
            cache_enabled=False,
        ):
            for index, block in enumerate(self.model.blocks):
                prefix = cache["layers"][index] if clean_length else None
                noisy_hidden = block(
                    noisy_hidden, noisy_rotary, noisy_condition, noisy_mask,
                    key_value_prefix=prefix
                )
            noisy_logits = self.model.project_hidden(noisy_hidden, noisy_condition)
        result = ModelPrediction(noisy_logits, noisy_hidden)
        return result if return_features else result.logits
