from .diffusion import Diffusion
from .algo_utils import (
    expected_rank, 
    sigma_from_time_vectorized, 
    inference_sigmas, 
    sample_categorical, 
    training_sigma_ve
    )

import utils
import torch
import torch.nn.functional as F
import math


class EncoderDecoder(Diffusion):
    def __init__(self, config, tokenizer):
        self.mask_index = len(tokenizer)
        vocab_size = len(tokenizer)
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self._validate_configuration()
        self.save_hyperparameters()

    def sample_t(self, n, accum_step):
        return super()._sample_t(n, accum_step)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def _validate_configuration(self):
        return

    def _process_model_output(self, model_output):
        model_output = model_output / self.temp
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        return model_output

    def nll_per_token(self,
                     log_x_theta,
                     x0_tokens):
        log_p_theta = log_x_theta.gather(dim=-1, index=x0_tokens.unsqueeze(-1)).squeeze(-1)
        return -log_p_theta

    def q_xt(self, x_clean, alpha_t):
        latent = self.backbone.encode(x_clean)
        return latent

    def forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs)
        return self._process_model_output(model_output=model_output)

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        latent = self.backbone.encode(x0)
        log_x_theta = self.backbone.decode(latent).log_softmax(dim=-1)
        return self.nll_per_token(log_x_theta, x0)

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name=k, value=v.compute(), on_step=False, on_epoch=True, sync_dist=True
            )
        self._train_mode()