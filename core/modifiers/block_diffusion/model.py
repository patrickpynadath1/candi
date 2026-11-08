"""Shared-transformer clean/noisy stream model wrapper."""

from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch import nn

from core.models.modular import ModularDiT
from core.models.components import DiffusionBlocks
from core.models.prediction import ModelPrediction
from core.models.kv_cache import append_key_value
from .layout import make_layout
from core.modifiers.encoder_decoder import EncoderDecoderModel


def view_stream_inputs(model, states, points, time, self_condition, clean, clean_encoder=None):
    """(noisy hidden, noisy condition, clean hidden, clean condition) for a ModularDiT."""
    noisy, noisy_condition = model.prepare(states, points, time, self_condition=self_condition)
    clean_hidden, clean_condition = model.clean_token_hidden(clean, clean_encoder)
    return noisy, noisy_condition, clean_hidden, clean_condition


class BlockDiffusionModel(nn.Module):
    def __init__(self, model, block_size, clean_encoder=None, x0_causal=True):
        super().__init__()
        input_model = model.model if isinstance(model, EncoderDecoderModel) else model
        if not isinstance(input_model, ModularDiT):
            raise TypeError(
                "block diffusion requires a model with fused-stream hidden interfaces"
            )
        if input_model.views is None and input_model.time.coordinate.startswith("negative_log_keep"):
            # The clean prefix of a single-state network runs on a Gaussian path
            # (time 0, sigma 0, alpha 1), where a keep-rate coordinate is undefined.
            raise TypeError(
                "block diffusion needs a network whose clean prefix has a time coordinate; "
                f"{input_model.time.coordinate} reads a keep rate"
            )
        self.model = model
        self.block_size = block_size
        self.x0_causal = x0_causal
        self.clean_encoder = clean_encoder
        self._training_block_size = ContextVar("training_block_size", default=None)
        self._clean = ContextVar("block_diffusion_clean", default=None)
        self.last_clean_logits = None
        self._parallel_clean = ContextVar("parallel_block_clean", default=None)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @contextmanager
    def block_size_context(self, size):
        token = self._training_block_size.set(size)
        try:
            yield
        finally:
            self._training_block_size.reset(token)

    @contextmanager
    def parallel_clean_context(self, tokens, blocks):
        context = self._parallel_clean.set((tokens, blocks))
        try:
            yield
        finally:
            self._parallel_clean.reset(context)

    @contextmanager
    def clean_context(self, tokens, *, offset=0, cache=None):
        if cache is not None:
            if self.training or torch.is_grad_enabled():
                cache = None
            else:
                size = self._training_block_size.get() or self.block_size
                previous = cache.get("prefix_tokens")
                if previous is not None:
                    old_length = previous.shape[1]
                    unchanged = (
                        previous.device == tokens.device
                        and previous.shape[0] == tokens.shape[0]
                        and old_length <= tokens.shape[1]
                        and torch.equal(previous, tokens[:, :old_length])
                    )
                    # Extending an incomplete block changes its bidirectional
                    # clean states, so it cannot be treated as an immutable prefix.
                    partial_extension = (
                        not self.x0_causal
                        and old_length % size
                        and old_length != tokens.shape[1]
                    )
                    if (
                        not unchanged
                        or partial_extension
                        or cache.get("block_size") != size
                    ):
                        cache.clear()
                cache["prefix_tokens"] = tokens.detach().clone()
                cache["block_size"] = size
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

    def forward(
        self, *inputs, self_condition=None, return_features=False, hidden_only=False
    ):
        current = self._clean.get()
        if current is None:
            if isinstance(self.model, EncoderDecoderModel):
                raise ValueError(
                    "encoder-decoder prediction requires a clean block context"
                )
            extra = {"hidden_only": True} if hidden_only else {}
            return self.model(
                *inputs,
                self_condition=self_condition,
                return_features=return_features,
                **extra,
            )
        # With a clean block context, hidden_only is only a saving: the block
        # paths still return the hidden state alongside logits.
        clean, offset, cache = current
        noisy = inputs[0][0] if isinstance(inputs[0], (tuple, list)) else inputs[0]
        length = noisy.shape[1]
        if len(clean) != len(noisy):
            raise ValueError("clean block context must match model batch")
        clean_length = clean.shape[1]
        layout = make_layout(
            length,
            self._training_block_size.get() or self.block_size,
            noisy.device,
            batch_size=len(noisy),
            offset=offset,
            clean_length=clean_length,
            x0_causal=self.x0_causal,
        )
        if isinstance(self.model, EncoderDecoderModel):
            clean_hidden, _, noisy_hidden, noisy_condition = self._stream_inputs(
                inputs, clean, self_condition
            )
            result = self.model(
                clean_hidden,
                noisy_hidden,
                noisy_condition,
                layout,
                cache=cache,
            )
            return result if return_features else result.logits
        # The cached path runs model.blocks directly, so it is only valid for
        # the standard policy (not token feedback or depth mixing).
        if (
            cache is not None
            and not self.training
            and type(self.model.block_execution) is DiffusionBlocks
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
        noisy_hidden = output.hidden[:, clean_length:]
        self.last_clean_logits = clean_logits
        auxiliary = output.auxiliary_logits
        if auxiliary is not None:
            auxiliary = auxiliary[:, clean_length:]
        result = ModelPrediction(noisy_logits, noisy_hidden, auxiliary_logits=auxiliary)
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
        noisy_hidden, noisy_condition, clean_hidden, clean_condition = view_stream_inputs(
            self.model, states, points, time, conditioning, clean, self.clean_encoder
        )
        boundary = head.left_boundary.to(noisy_hidden).expand(len(noisy_hidden), 1, -1)
        noisy_hidden = torch.cat([boundary, noisy_hidden], 1)
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
        noisy, noisy_condition, clean_hidden, clean_condition = view_stream_inputs(
            self.model, states, points, time, conditioning, clean, self.clean_encoder
        )
        boundary = head.left_boundary.to(noisy).expand(len(noisy), 1, -1)
        noisy = torch.cat([boundary, noisy], 1)
        width = noisy.shape[1]
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
        model = (
            self.model.model
            if isinstance(self.model, EncoderDecoderModel)
            else self.model
        )
        states, points, global_time = inputs
        noisy_hidden, noisy_condition, clean_hidden, clean_condition = view_stream_inputs(
            model, states, points, global_time, self_condition, clean, self.clean_encoder
        )
        return clean_hidden, clean_condition, noisy_hidden, noisy_condition

    def _cached_forward(
        self, inputs, clean, offset, cache, layout, self_condition, return_features
    ):
        clean_length = clean.shape[1]
        encoded_length = cache.get("clean_length", 0)
        clean_hidden, clean_condition, noisy_hidden, noisy_condition = (
            self._stream_inputs(inputs, clean[:, encoded_length:], self_condition)
        )
        layers = cache.setdefault("layers", [None] * len(self.model.blocks))
        if clean_length > encoded_length:
            clean_ids = layout.position_ids[:, encoded_length:clean_length]
            with torch.autocast(clean.device.type, enabled=False):
                clean_rotary = self.model.rotary_emb(
                    clean_hidden, position_ids=clean_ids
                )
            causal = layout.attention_mask[
                :, :, encoded_length:clean_length, :clean_length
            ]
            with torch.autocast(
                clean.device.type,
                enabled=torch.is_autocast_enabled(clean.device.type),
                cache_enabled=False,
            ):
                for index, block in enumerate(self.model.blocks):
                    clean_hidden, new_kv = block(
                        clean_hidden,
                        clean_rotary,
                        clean_condition,
                        causal,
                        key_value_prefix=layers[index],
                        return_key_value=True,
                    )
                    layers[index] = append_key_value(layers[index], new_kv)
            cache["clean_length"] = clean_length
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
