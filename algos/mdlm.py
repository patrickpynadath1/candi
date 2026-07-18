from .absorbing_state import AbsorbingState
from .registry import algo_registry

import torch

@algo_registry.register("mdlm")
class MDLM(AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.config.sampling.predictor == "ancestral_cache"

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        model_output = model_output / self.temp
        model_output[:, :, self.mask_index] += self.neg_infinity

        # Normalize the model_output such that x.exp() is
        # a probability distribution over vocab_size.
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)


        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(
            unmasked_score, -1, x[..., None], torch.zeros_like(unmasked_score[..., :1])
        )
        unmasked_score[:, :, self.mask_index] = -(log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
        model_output = masked_score * masked_indices + unmasked_score * (
            1 - masked_indices
        )
        return model_output.exp()

    def validation_step(self, batch, batch_idx):

        del batch_idx
        losses = self._loss(batch["input_ids"], batch["attention_mask"])
        # t, mean_nlls = self.unweighted_nlls(batch["input_ids"], batch["attention_mask"])
        # if self.ce_time_pairs['t'] is None:
        #     self.ce_time_pairs['t'] = t.cpu()
        #     self.ce_time_pairs['ce'] = mean_nlls.cpu()
        # else: 
        #     self.ce_time_pairs['t'] = torch.cat((self.ce_time_pairs['t'], t.cpu()), dim=0)
        #     self.ce_time_pairs['ce'] = torch.cat((self.ce_time_pairs['ce'], mean_nlls.cpu()), dim=0)
        self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
        return losses.loss
    

    def unweighted_nlls(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]


        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)

        log_x_theta = self.forward(xt, sigma=sigma)
        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)
        sum_nll = -1 * log_p_theta.sum(dim=-1)
        valid_tokens = valid_tokens.sum(dim=-1)
        mean_nlls = sum_nll / valid_tokens
        return t, mean_nlls
