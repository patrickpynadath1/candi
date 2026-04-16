from .cosmos import EncNormalizer, Encoder, Decoder


import torch.nn as nn
import huggingface_hub
from .registry import models


@models.register("encoder_decoder_cosmos")
class EncoderDecoderCosmos(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size): 
        super().__init__()

        self.config = config 
        self.vocab_size = vocab_size 

        encoder_cfg = config.model.encoder
        encoder_cfg.tokens.vocab_size = vocab_size
        decoder_cfg = config.model.decoder
        decoder_cfg.tokens.vocab_size = vocab_size
        decoder_cfg.tokens.mask_token_id = vocab_size-1

        self.encoder = Encoder(encoder_cfg)
        self.enc_normalizer = EncNormalizer(encoder_cfg)
        self.decoder = Decoder(decoder_cfg)

    def forward(self, input_ids, attention_mask, token_embeddings=None):
        encoder_latents = self.encoder(
            token_ids=input_ids,
            mask_tokens=attention_mask,
            token_embeddings=token_embeddings,
        )
        encoder_latents = self.enc_normalizer(encoder_latents)
        decoder_output = self.decoder(encoder_latents, masked_input_ids=None, return_last_hidden_state=False)
        return decoder_output
