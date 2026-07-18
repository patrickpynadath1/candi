"""Flow Language Model training objective and sampler-facing API."""

import torch
import torch.nn.functional as F

from .flm_utils import alpha_to_gamma, build_luts, gamma_to_alpha
from .registry import algo_registry
from .trainer_base import TrainerBase


@algo_registry.register("flm")
class FLM(TrainerBase):
    default_sampler = "flm_euler"
    supported_samplers = frozenset({"flm_euler"})
    sampling_family = "flm"
    sampling_capabilities = frozenset({"flm_prior", "flm_time_map", "flm_logits"})

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.cap_value = getattr(config.algo, "cap_value", None)
        a2g, g2a = build_luts(self.vocab_size)
        self.register_buffer("lut_a2g", a2g)
        self.register_buffer("lut_g2a", g2a)

    def _validate_configuration(self):
        pass

    def _process_model_input(self, x0, valid):
        return x0, None, valid

    def _process_sigma(self, sigma):
        sigma = sigma.squeeze(-1) if sigma.ndim == 2 else sigma
        return sigma.reshape(1) if sigma.ndim == 0 else sigma

    def _process_model_output(self, model_output, xt=None, sigma=None):
        del xt, sigma
        if self.cap_value is not None:
            model_output = self.cap_value * torch.tanh(model_output / self.cap_value)
        return model_output.log_softmax(-1)

    def forward(self, xt, sigma):
        sigma = self._process_sigma(sigma)
        return self._process_model_output(self.backbone(xt, sigma))

    def alpha_to_gamma(self, alpha):
        return alpha_to_gamma(alpha, self.lut_a2g)

    def gamma_to_alpha(self, gamma):
        return gamma_to_alpha(gamma, self.lut_g2a)

    def sample_t(self, n, accum=None):
        del accum
        u = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            u = (u / n + torch.arange(n, device=self.device) / n) % 1
        return self.t_min + (self.t_max - self.t_min) * u

    def corrupt(self, x0, gamma, valid_tokens=None):
        target = F.one_hot(x0, self.vocab_size).float()
        g = gamma[:, None, None]
        xt = (1 - g) * torch.randn_like(target) + g * target
        if valid_tokens is not None:
            xt = torch.where(valid_tokens.bool().unsqueeze(-1), xt, target)
        return xt, target

    def nll(
        self,
        input_tokens,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
    ):
        del output_tokens, train_mode
        t = self.sample_t(input_tokens.shape[0], current_accumulation_step)
        gamma = self.alpha_to_gamma(t)
        xt, target = self.corrupt(input_tokens, gamma)
        loss = -(target * self.forward(xt, t)).sum(-1)
        weighting = getattr(self.backbone, "learnable_loss_weighting", None)
        if weighting is not None:
            w = weighting(t).unsqueeze(-1)
            loss = torch.exp(-w) * loss + w
        return loss

    def prior_sample(self, num_samples, length):
        return torch.randn(num_samples, length, self.vocab_size, device=self.device)

    def sampling_gamma(self, t):
        return self.alpha_to_gamma(t)

    def sampling_logits(self, x, t):
        return self.forward(x, t)
