from copy import deepcopy

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn

from .common import EmbeddingLayer, LatentBottleneckDecoder, LatentBottleneckEncoder
from .cosmos import EncNormalizer
from .registry import models


torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def _to_config(config):
    if isinstance(config, dict):
        return omegaconf.OmegaConf.create(config)
    return config


class NativeCosmosEncoder(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = _to_config(config)
        self.vocab_size = vocab_size

        dim = self.config.embedding.dim
        self.dim = dim
        self.num_latents = self.config.latent.num_latents

        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.backbone = LatentBottleneckEncoder(
            embedding_dim=self.config.embedding.dim,
            latent_dim=self.config.latent.dim,
            hidden_size=self.config.hidden.size,
            num_layers=self.config.hidden.num_layers,
            num_latents=self.config.latent.num_latents,
            num_heads=self.config.attention.num_heads,
            head_dim=self.config.attention.head_size,
            max_position_embeddings=self.config.embedding.max_position_embeddings,
            ff_mult=self.config.hidden.ff_mult,
            dropout=self.config.hidden.dropout,
            layer_eps=self.config.normalization.layer_eps,
            qk_norm=self.config.attention.qk_norm,
            initializer_range=self.config.embedding.initializer_range,
        )

    def forward(
        self,
        input_tokens,
        sigma=None,
        attention_mask=None,
        token_ids=None,
        mask_tokens=None,
        token_embeddings=None,
        skip_embedding=False,
        **kwargs,
    ):
        del sigma, kwargs

        if token_ids is None:
            token_ids = input_tokens

        if mask_tokens is None:
            mask_tokens = attention_mask

        if skip_embedding:
            token_embeddings = input_tokens
        elif token_embeddings is None:
            token_embeddings = self.vocab_embed(token_ids)

        return self.backbone(token_embeddings=token_embeddings, mask_tokens=mask_tokens)

    def get_embedding(self, x):
        return self.vocab_embed(x)


class NativeCosmosDecoder(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = _to_config(config)
        self.vocab_size = vocab_size

        self.dim = self.config.embedding.dim
        self.mask_index = self.config.tokens.mask_token_id

        self.backbone = LatentBottleneckDecoder(
            embedding_dim=self.config.embedding.dim,
            latent_dim=self.config.latent.dim,
            hidden_size=self.config.hidden.size,
            num_layers=self.config.hidden.num_layers,
            num_latents=self.config.latent.num_latents,
            num_heads=self.config.attention.num_heads,
            head_dim=self.config.attention.head_size,
            vocab_size=vocab_size,
            max_seq_len=self.config.embedding.max_position_embeddings,
            mask_token_id=self.config.tokens.mask_token_id,
            ff_mult=self.config.hidden.ff_mult,
            dropout=self.config.hidden.dropout,
            layer_eps=self.config.normalization.layer_eps,
            qk_norm=self.config.attention.qk_norm,
            initializer_range=self.config.embedding.initializer_range,
        )

    def initialize_embedding_from_table(self, embedding_table, freeze=None):
        del freeze
        self.backbone.initialize_embedding_from_table(embedding_table)

    def forward(
        self,
        input_latent,
        sigma=None,
        generation_mode=False,
        masked_input_ids=None,
        return_last_hidden_state=False,
        return_hr_vector=False,
        **kwargs,
    ):
        del sigma, generation_mode, kwargs

        output = self.backbone(
            encoder_latents=input_latent,
            masked_input_ids=masked_input_ids,
            return_last_hidden_state=return_last_hidden_state or return_hr_vector,
        )
        if return_hr_vector:
            logits, hidden = output
            return hidden, logits
        return output


@models.register("encoder_decoder_cosmos_ntv")
class EncoderDecoderCosmosNative(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = _to_config(config)
        self.vocab_size = vocab_size
        self.mask_index = vocab_size - 1

        if self.config.algo.use_raw_tokens:
            self.token_embedding = nn.Embedding(
                vocab_size,
                self.config.model.encoder.embedding.dim,
            )

        encoder_cfg = deepcopy(self.config.model.encoder)
        decoder_cfg = deepcopy(self.config.model.decoder)
        encoder_cfg.tokens.vocab_size = vocab_size
        decoder_cfg.tokens.vocab_size = vocab_size
        decoder_cfg.tokens.mask_token_id = self.mask_index

        self.encoder = NativeCosmosEncoder(encoder_cfg, vocab_size)
        self.enc_normalizer = EncNormalizer(encoder_cfg)
        self.decoder = NativeCosmosDecoder(decoder_cfg, vocab_size)

    def encode(self, **kwargs):
        return self.encoder(**kwargs)

    def decode(self, **kwargs):
        return self.decoder(**kwargs)

    def get_embedding(self, x):
        return self.encoder.get_embedding(x)

    def forward(self, x, sigma=None, return_latents=False, attention_mask=None, token_embeddings=None):
        latents = self.encoder(
            input_tokens=x,
            sigma=sigma,
            attention_mask=attention_mask,
            token_embeddings=token_embeddings,
        )
        logits = self.decoder(input_latent=latents, sigma=sigma)
        if return_latents:
            return latents, logits
        return logits
