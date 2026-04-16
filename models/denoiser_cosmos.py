from .cosmos import TransformerEncoder, ConditionalEncoder, timestep_embedding
from .registry import models

import torch
import torch.nn as nn

@models.register("cosmos_denoiser")
class ScoreEstimator(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(ScoreEstimator, self).__init__()

        self.config = config
        self.hidden_size = config.algo.architecture.unconditional_encoder.hidden_size
        self.time_max_period = config.algo.architecture.time_embedding.max_period
        self.is_conditional = config.algo.diffusion.is_conditional


        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        )

        self.encoder = TransformerEncoder(config)

        if self.is_conditional:
            self.conditional_encoder = ConditionalEncoder(config)

        if self.is_conditional:
            self._max_position_embeddings = self.config.algo.architecture.conditional_encoder.max_position_embeddings
            self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
            self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self.hidden_size)

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: torch.Tensor,
            cond=None,
            attention_mask=None,
            cond_mask=None,
            x_0_self_cond=None,
    ):
        hidden_t = self.time_emb(timestep_embedding(time_t, self.hidden_size, self.time_max_period))

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=x_t.dtype
            )
            
        if cond_mask is not None:
            cond_mask = self.get_extended_attention_mask(
                attention_mask=cond_mask,
                dtype=x_t.dtype
            )
        
        if self.is_conditional:
            cond = self.conditional_encoder(
                hidden_states=cond,
                attention_mask=cond_mask
            )

            seq_length = cond.size(1)
            position_ids = self.position_ids[:, : seq_length]
            emb_pos = self.position_embeddings(position_ids)
            cond = cond + emb_pos

        output = self.encoder(
            x=x_t,
            attention_mask=attention_mask,
            emb_t=hidden_t,
            cond=cond,
            cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond,
        )
        return output
