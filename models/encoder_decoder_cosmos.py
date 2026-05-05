from .cosmos import EncNormalizer, Encoder, Decoder


import torch
import torch.nn as nn
import torch.nn.functional as F
import huggingface_hub
from .registry import models


@models.register("encoder_decoder_cosmos")
class EncoderDecoderCosmos(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size): 
        super().__init__()

        self.config = config 
        self.vocab_size = vocab_size 
        self.encoder_embedding_dim = config.model.encoder.embedding.dim
        self.aux_hidden_dim = config.model.encoder.get(
            "aux_hidden_dim", self.encoder_embedding_dim
        )

        if self.config.algo.use_raw_tokens:
            self.token_embedding = nn.Embedding(vocab_size, config.model.encoder.hidden_size)
        if self.aux_hidden_dim == self.encoder_embedding_dim:
            self.aux_hidden_projection = nn.Identity()
        else:
            self.aux_hidden_projection = nn.Linear(
                self.aux_hidden_dim,
                self.encoder_embedding_dim,
                bias=False,
            )

        encoder_cfg = config.model.encoder
        encoder_cfg.tokens.vocab_size = vocab_size
        decoder_cfg = config.model.decoder
        decoder_cfg.tokens.vocab_size = vocab_size
        decoder_cfg.tokens.mask_token_id = vocab_size-1

        self.encoder = Encoder(encoder_cfg)
        self.enc_normalizer = EncNormalizer(encoder_cfg)
        self.decoder = Decoder(decoder_cfg)

    def project_aux_features(self, aux_features):
        if aux_features is None or aux_features.shape[-1] == self.encoder_embedding_dim:
            return aux_features
        return self.aux_hidden_projection(aux_features)

    def project_aux_embedding_table(self, embedding_table):
        if isinstance(embedding_table, nn.Embedding):
            weight = embedding_table.weight
        elif isinstance(embedding_table, nn.Parameter):
            weight = embedding_table
        elif torch.is_tensor(embedding_table):
            weight = embedding_table
        elif hasattr(embedding_table, "embedding") and torch.is_tensor(embedding_table.embedding):
            weight = embedding_table.embedding
        else:
            raise TypeError("Unsupported embedding table type for auxiliary projection.")

        return self.project_aux_features(weight)

    def reconstruct_aux_features(self, model_features):
        if model_features is None or self.aux_hidden_dim == self.encoder_embedding_dim:
            return model_features
        return F.linear(model_features, self.aux_hidden_projection.weight.t())

    def forward(self, input_ids, attention_mask, token_embeddings=None):
        token_embeddings = self.project_aux_features(token_embeddings)
        encoder_latents = self.encoder(
            token_ids=input_ids,
            mask_tokens=attention_mask,
            token_embeddings=token_embeddings,
        )
        encoder_latents = self.enc_normalizer(encoder_latents)
        decoder_output = self.decoder(encoder_latents, masked_input_ids=None, return_last_hidden_state=False)
        return decoder_output
