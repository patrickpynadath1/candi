from .diffusion import Diffusion
from .algo_utils import (
    expected_rank, 
    sigma_from_time_vectorized, 
    inference_sigmas, 
    sample_categorical, 
    training_sigma_ve
    )
from .registry import algo_registry
import utils
import torch
import torch.nn.functional as F


@algo_registry.register("cadd")
class CADD(Diffusion):
    def __init__(self, config, tokenizer):
        self.mask_index = len(tokenizer)
        vocab_size = len(tokenizer) + 1
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self._validate_configuration()
        self.save_hyperparameters()
        self.temp = 1.0


    def sample_t(self, n, accum_step):
        return super()._sample_t(n, accum_step)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def _validate_configuration(self):
        return

    def _process_model_output(self, xt_tokens, model_output, reveal_mask):
        model_output = model_output / self.temp
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        reveal_mask = reveal_mask.bool()

        # for pure continuous diffusion, reveal mask is all false -- so this doesnt have any impact
        model_output[reveal_mask] = self.neg_infinity
        model_output[reveal_mask, xt_tokens[reveal_mask]] = 0
        return model_output

    def nll_per_token(self,
                     log_x_theta,
                     x0_tokens,
                     alpha_t,
                     dalpha_t,
                     reveal_mask,
                     **kwargs):


        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0_tokens[:, :, None]
        ).squeeze(-1)
        nll = log_p_theta * dalpha_t / (1 - alpha_t)
        return nll


    def discrete_noising(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape
            (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        xt = torch.where(move_indices, self.mask_index, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def q_xt(self, x, alpha_t):

        disc_xt = self.discrete_noising(x, alpha_t)
        reveal_mask = (disc_xt == x).float()
        discrete_noise = (1 - alpha_t).squeeze()
        clean_embeds = self.backbone.get_embedding(x)
        noise = torch.randn_like(clean_embeds)
        noisy_embeds = alpha_t[:, :, None] * clean_embeds + (1 - alpha_t)[:, :, None] * noise
        mask_embed = self.backbone.get_mask_embed()
        noisy_embeds = noisy_embeds + mask_embed[None, None, :]
        xt = torch.where(reveal_mask[:, :, None] == 1, clean_embeds, noisy_embeds)
        return {
            'xt': xt,
            'reveal_mask': reveal_mask,
            'discrete_noise': discrete_noise,
            'xt_tokens': disc_xt,
        }

    def prior_sample(self, *batch_dims):
        noise = torch.randn(
            *batch_dims, self.config.model.hidden_size, dtype=torch.float32, device=self.device
        ) 
        return noise

    def forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs, precond=False, is_embed=True)
        return self._process_model_output(model_output=model_output, xt_tokens=kwargs['xt_tokens'], reveal_mask=kwargs['reveal_mask'])

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2

        noisy_output = self.q_xt(x0, alpha_t)
        log_x_theta = self.forward(**noisy_output)

        utils.print_nans(log_x_theta, "model_output")
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            x0_tokens=x0,
            low_var=train_mode,
            **noisy_output,
        )


    def generate_sample_prompt(self, prompt_tokens, x, prompt_mask):
        prompt_onehots = torch.nn.functional.one_hot(prompt_tokens, x.size(-1)).float().to(x.device)
        prompt_mask = prompt_mask.bool()
        x = torch.where(prompt_mask.unsqueeze(-1), prompt_onehots, x)
        return x, prompt_mask

     # optimized function with cached embeddings
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

        xt_embeds = self.prior_sample(num_samples, self.num_tokens)
        if num_steps is None:
            num_steps = self.config.sampling.steps

        clean_mask = torch.zeros((num_samples, self.num_tokens), device=xt_embeds.device, dtype=torch.bool)
        if prompt_tokens is not None and prompt_mask is not None:
            xt_embeds, clean_mask = self.generate_sample_prompt(prompt_tokens, xt_embeds, prompt_mask)
        timesteps = torch.linspace(.999, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / (num_steps)
        xt_tokens = torch.full((num_samples, self.num_tokens), self.mask_index, device=xt_embeds.device, dtype=torch.long)
        if prompt_tokens is not None and prompt_mask is not None:
            xt_tokens = torch.where(prompt_mask.bool(), prompt_tokens, xt_tokens)
        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i+1]

            xt_embeds, x0_hat_tokens = self._continuous_step(xt_embeds=xt_embeds,
                                                            xt_tokens=xt_tokens, 
                                                            time_t=t, 
                                                            time_s=s, 
                                                            clean_mask=clean_mask)

            xt_embeds, xt_tokens, clean_mask = self._discrete_step_optimized(x0_hat_tokens=x0_hat_tokens,
                                                                            xt_tokens=xt_tokens,
                                                                            xt_embeds=xt_embeds,
                                                                            t=t,
                                                                            dt=dt,
                                                                            prev_clean_mask=clean_mask,
                                                                            noise_removal_step=(i==num_steps-1))
        return xt_tokens

    def _discrete_step_optimized(self, x0_hat_tokens, xt_tokens, xt_embeds, t, dt, prev_clean_mask, noise_removal_step=False):

        if noise_removal_step:
            s=0
        else:
            s = t - dt

        # unmasking correspends to 1-alpha(s) / 1-alpha(t)
        # this is just s / t under a log linear schedule
        unmask = torch.rand(prev_clean_mask.shape, device=prev_clean_mask.device) < (t-s)/t

        # carry over clean tokens from previous step, and unmask some additional tokens according to noise schedule
        xt_tokens[~prev_clean_mask] = x0_hat_tokens[~prev_clean_mask]
        new_clean_mask = prev_clean_mask | unmask
        return xt_embeds, xt_tokens, new_clean_mask

    @torch.no_grad()
    def _continuous_step(
        self,
        xt_embeds: torch.Tensor,
        xt_tokens: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        clean_mask: torch.Tensor=None,
    ) -> torch.Tensor:

        dt_vec = torch.ones(xt_embeds.shape[0], device=xt_embeds.device) * (time_t - time_s).abs().item()
        time_t_vec = torch.ones(xt_embeds.shape[0], device=xt_embeds.device) * time_t.item()

        if clean_mask is None:
            clean_mask = torch.zeros(xt_embeds.shape[:-1], device=xt_embeds.device)
        cond_denoised = self.forward(xt=xt_embeds, 
                                     xt_tokens=xt_tokens, 
                                     discrete_noise=time_t_vec, 
                                     reveal_mask=clean_mask,
                                     ).double()
        denoised = cond_denoised.exp()
        x0_hat_tokens = sample_categorical(denoised)
        x0_hat = self.backbone.get_embedding(x0_hat_tokens)
        xt_embeds_new = xt_embeds + (dt_vec / (1 - time_t_vec))[:, None, None] * (x0_hat - xt_embeds)
        xt_embeds_new[clean_mask] = xt_embeds[clean_mask]
        return xt_embeds_new, x0_hat_tokens
