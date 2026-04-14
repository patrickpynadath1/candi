# torch.serialization.add_safe_globals([DictConfig])


from .trainer_base import TrainerBase
from .registry import algo_registry

import torch


@algo_registry.register("ar")
class AR(TrainerBase):
    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()
        self.temp = config.algo.get("temp", 1.0)

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == "none"

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(self, input_tokens, output_tokens, current_accumulation_step, *args):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_index] = self.neg_infinity
        output = output.log_softmax(-1)
        return -output.gather(-1, output_tokens[:, :, None])[:, :, 0]

    def generate_samples(self, num_samples, prompt_mask=None, prompt_tokens=None, **kwargs):
        # precompute token buffer
        num_pred_tokens = self.num_tokens - 1
        x = torch.zeros(
            (num_samples, num_pred_tokens + 1), dtype=torch.long, device=self.device
        )

        if prompt_mask is not None and prompt_tokens is not None:
            x = (prompt_mask * prompt_tokens + (1 - prompt_mask) * x).long()
            start = prompt_mask.sum(1).max().long().item() - 1
        else:
            start = 0
            x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (
            torch.distributions.Gumbel(0, 1)
            .sample((num_samples, num_pred_tokens, self.vocab_size))
            .to(self.device)
        )
        if self.config.sampling.use_float64:
            noise = noise.to(torch.float64)
        for i in range(start, num_pred_tokens):
            output = self.backbone(x[:, : i + 1], None)
            output[:, :, self.mask_index] = self.neg_infinity
            output = output / self.temp
            output = output.log_softmax(-1)
            y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _process_sigma(self, sigma):
        del sigma
        return None