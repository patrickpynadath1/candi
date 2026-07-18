# torch.serialization.add_safe_globals([DictConfig])


from .trainer_base import TrainerBase
from .registry import algo_registry

@algo_registry.register("ar")
class AR(TrainerBase):
    default_sampler = "ar"
    supported_samplers = frozenset({"ar"})
    sampling_family = "autoregressive"
    sampling_capabilities = frozenset({"next_token_logits"})
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

    def next_token_logits(self, tokens):
        logits = self.backbone(tokens, None)
        logits[:, :, self.mask_index] = self.neg_infinity
        return logits

    def _process_sigma(self, sigma):
        del sigma
        return None
