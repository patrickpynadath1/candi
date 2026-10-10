"""Shared-transformer clean/noisy stream model wrapper."""

from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch import nn

from core.models.continuous_dit import ContinuousDiT
from core.models.dit import DiT
from core.modifiers.hybrid.model import HybridModel
from core.models.prediction import ModelPrediction
from .layout import make_layout


class BlockDiffusionModel(nn.Module):
    def __init__(self, model, block_size, clean_encoder=None, x0_causal=True):
        super().__init__()
        if not isinstance(model, (DiT, ContinuousDiT, HybridModel)):
            raise TypeError(
                "block diffusion requires a model with fused-stream hidden interfaces"
            )
        self.model = model
        self.block_size = block_size
        self.x0_causal = x0_causal
        self.clean_encoder = clean_encoder
        self._clean = ContextVar("block_diffusion_clean", default=None)
        self.last_clean_logits = None
        self._parallel_clean = ContextVar("parallel_block_clean", default=None)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @contextmanager
    def parallel_clean_context(self, tokens, blocks):
        context = self._parallel_clean.set((tokens, blocks))
        try:
            yield
        finally:
            self._parallel_clean.reset(context)

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
        clean_hidden, noisy_hidden = (
            output.hidden[:, :length],
            output.hidden[:, length:],
        )
        clean_logits, noisy_logits = (
            output.logits[:, :length],
            output.logits[:, length:],
        )
        self.last_clean_logits = clean_logits
        auxiliary = output.auxiliary_logits
        if auxiliary is not None:
            auxiliary = auxiliary[:, length:]
        return ModelPrediction(
            noisy_logits, noisy_hidden, clean_logits, clean_hidden, auxiliary
        )

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
            length,
            self.block_size,
            noisy.device,
            batch_size=len(inputs[0]),
            offset=offset,
            clean_length=clean_length,
            x0_causal=self.x0_causal,
        )
        if (
            cache is not None
            and not self.training
            and not getattr(
                self.model.block_execution, "uses_intermediate_feedback", False
            )
        ):
            return self._cached_forward(
                inputs, clean, offset, cache, layout, self_condition, return_features
            )
        clean_hidden, clean_condition, noisy_hidden, noisy_condition = (
            self._stream_inputs(inputs, clean, self_condition)
        )
        condition = None
        if clean_condition is not None:
            condition = torch.cat(
                [
                    clean_condition[:, None].expand(-1, clean_length, -1),
                    noisy_condition[:, None].expand(-1, length, -1),
                ],
                1,
            )
        output = self.model.forward_hidden(
            torch.cat([clean_hidden, noisy_hidden], 1),
            condition,
            return_features=True,
            attention_mask=layout.attention_mask,
            position_ids=layout.position_ids,
        )
        clean_logits, noisy_logits = (
            output.logits[:, :clean_length],
            output.logits[:, clean_length:],
        )
        clean_hidden, noisy_hidden = (
            output.hidden[:, :clean_length],
            output.hidden[:, clean_length:],
        )
        self.last_clean_logits = clean_logits
        auxiliary = output.auxiliary_logits
        if auxiliary is not None:
            auxiliary = auxiliary[:, clean_length:]
        result = ModelPrediction(
            noisy_logits, noisy_hidden, clean_logits, clean_hidden, auxiliary
        )
        return result if return_features else result.logits

    def forward_sequence_features(
        self, head, states, points, time, valid, conditioning=None
    ):
        """Variable-length feature path with a completed clean prefix.

        The gap head's left boundary belongs to the noisy stream. Completed
        blocks are re-encoded as clean tokens on every call; this deliberately
        avoids reusing KV entries after a preceding block changed length.
        """
        parallel = self._parallel_clean.get()
        if parallel is not None:
            return self._parallel_sequence_features(
                head, states, points, time, valid, conditioning, *parallel
            )
        current = self._clean.get()
        if current is None:
            return None
        clean, offset, _cache = current
        noisy_hidden, noisy_condition = self.model.prepare_fused(
            states, points, time, self_condition=conditioning
        )
        boundary = head.left_boundary.to(noisy_hidden).expand(len(noisy_hidden), 1, -1)
        noisy_hidden = torch.cat([boundary, noisy_hidden], 1)
        probabilities = torch.nn.functional.one_hot(
            clean,
            self.model.output_vocab_size or self.model.output_layer.linear.out_features,
        ).float()
        clean_hidden, clean_condition = self.model.clean_state_hidden(probabilities)
        clean_length, noisy_length = clean.shape[1], noisy_hidden.shape[1]
        condition = None
        if clean_condition is not None:
            condition = torch.cat(
                [
                    clean_condition[:, None].expand(-1, clean_length, -1),
                    noisy_condition[:, None].expand(-1, noisy_length, -1),
                ],
                1,
            )
        # Logical block identity is explicit here: every clean token belongs to
        # a completed block and every noisy position belongs to the active one,
        # even when earlier insertions changed physical prefix length.
        clean_causal = torch.ones(
            clean_length,
            clean_length,
            dtype=torch.bool,
            device=noisy_hidden.device,
        ).tril()
        visible = torch.cat(
            [
                torch.cat(
                    [
                        clean_causal,
                        torch.zeros(
                            clean_length,
                            noisy_length,
                            dtype=torch.bool,
                            device=noisy_hidden.device,
                        ),
                    ],
                    1,
                ),
                torch.ones(
                    noisy_length,
                    clean_length + noisy_length,
                    dtype=torch.bool,
                    device=noisy_hidden.device,
                ),
            ],
            0,
        )[None, None].expand(len(clean), 1, -1, -1)
        key_valid = torch.cat(
            [
                torch.ones(
                    len(clean),
                    clean_length + 1,
                    dtype=torch.bool,
                    device=clean.device,
                ),
                valid,
            ],
            1,
        )
        visible = visible & key_valid[:, None, None, :]
        positions = torch.cat(
            [
                torch.arange(clean_length, device=clean.device),
                torch.arange(offset, offset + noisy_length, device=clean.device),
            ]
        )[None].expand(len(clean), -1)
        output = self.model.forward_hidden(
            torch.cat([clean_hidden, noisy_hidden], 1),
            condition,
            return_features=True,
            attention_mask=visible,
            position_ids=positions,
        )
        active = output.hidden[:, clean_length:]
        expected, gap_logits = head(active, valid)
        logits = output.logits[:, clean_length + 1 :, : head.vocab_size]
        return ModelPrediction(logits, active[:, 1:]), expected, gap_logits

    def _parallel_sequence_features(
        self, head, states, points, time, valid, conditioning, clean, blocks
    ):
        batch, clean_length = clean.shape
        noisy, noisy_condition = self.model.prepare_fused(
            states, points, time, self_condition=conditioning
        )
        boundary = head.left_boundary.to(noisy).expand(len(noisy), 1, -1)
        noisy = torch.cat([boundary, noisy], 1)
        width = noisy.shape[1]
        probabilities = torch.nn.functional.one_hot(
            clean,
            self.model.output_vocab_size or self.model.output_layer.linear.out_features,
        ).float()
        clean_hidden, clean_condition = self.model.clean_state_hidden(probabilities)
        hidden = torch.cat([clean_hidden, noisy.reshape(batch, blocks * width, -1)], 1)
        condition = None
        if clean_condition is not None:
            condition = torch.cat(
                [
                    clean_condition[:, None].expand(-1, clean_length, -1),
                    noisy_condition[:, None]
                    .expand(-1, width, -1)
                    .reshape(batch, blocks * width, -1),
                ],
                1,
            )
        device = clean.device
        clean_pos = torch.arange(clean_length, device=device)
        block_ids = torch.arange(blocks, device=device).repeat_interleave(width)
        starts = block_ids * self.block_size
        visible = torch.cat(
            [
                torch.cat(
                    [
                        clean_pos[None, :] <= clean_pos[:, None],
                        torch.zeros(
                            clean_length,
                            blocks * width,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    1,
                ),
                torch.cat(
                    [
                        clean_pos[None, :] < starts[:, None],
                        block_ids[:, None] == block_ids[None, :],
                    ],
                    1,
                ),
            ],
            0,
        )[None, None]
        noisy_valid = torch.nn.functional.pad(valid, (1, 0), value=True).reshape(
            batch, -1
        )
        keys = torch.cat([torch.ones_like(clean, dtype=torch.bool), noisy_valid], 1)
        visible = visible & keys[:, None, None, :]
        noisy_pos = starts + torch.arange(width, device=device).repeat(blocks)
        positions = torch.cat([clean_pos, noisy_pos])[None].expand(batch, -1)
        output = self.model.forward_hidden(
            hidden,
            condition,
            return_features=True,
            attention_mask=visible,
            position_ids=positions,
        )
        active = output.hidden[:, clean_length:].reshape(batch * blocks, width, -1)
        logits = output.logits[:, clean_length:].reshape(batch * blocks, width, -1)
        expected, gap_logits = head(active, valid)
        return (
            ModelPrediction(logits[:, 1:, : head.vocab_size], active[:, 1:]),
            expected,
            gap_logits,
        )

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
                )
                if self.model.adaLN
                else None
            )
        return clean_hidden, clean_condition, noisy_hidden, noisy_condition

    def _cached_forward(
        self, inputs, clean, offset, cache, layout, self_condition, return_features
    ):
        clean_hidden, clean_condition, noisy_hidden, noisy_condition = (
            self._stream_inputs(inputs, clean, self_condition)
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
                            clean_hidden,
                            clean_rotary,
                            clean_condition,
                            causal,
                            return_key_value=True,
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
                    noisy_hidden,
                    noisy_rotary,
                    noisy_condition,
                    noisy_mask,
                    key_value_prefix=prefix,
                )
            noisy_logits = self.model.project_hidden(noisy_hidden, noisy_condition)
        result = ModelPrediction(noisy_logits, noisy_hidden)
        return result if return_features else result.logits
