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


class ContDiffEmbed(Diffusion):
    def __init__(self, config, tokenizer):
        self.mask_index = len(tokenizer)

        vocab_size = len(tokenizer)
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self._validate_configuration()
        self.save_hyperparameters()

        self.pure_continuous = config.algo.pure_continuous

        self.continuous_noise =torch.linspace(.2, 4.0, 1000)
        self.discrete_noise = expected_rank(d=self.vocab_size, a=torch.tensor(1).to(self.device), sigma=self.continuous_noise)

        self.continuous_noise = torch.tensor(self.continuous_noise)
        self.discrete_noise = torch.tensor(self.discrete_noise)

        self.step_size = config.algo.step_size

        self.min_percentile = config.algo.min_percentile
        self.max_percentile = config.algo.max_percentile

        self.sigma_min = config.algo.sigma_min
        self.sigma_max = config.algo.sigma_max
        self.temp = config.algo.temp
        self.sampler = config.algo.sampler

    def ve_sde_noise_sched(self, t, sigma_min=0.01, sigma_max=10.0):
        return sigma_min * (sigma_max / sigma_min) ** t

    def sample_t(self, n, accum_step):
        return super()._sample_t(n, accum_step)

    def get_continuous_from_discrete_noise(self, discrete_noise):
        sigmas = training_sigma_ve(discrete_noise, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        return sigmas

    # inference schedule 
    # while the embedding diffusion should use the VE schedule, this produced better results 
    # at least on text8 -- both schedules for embedding diffusion collapsed on OWT
    def get_continuous_noise_sched_pure_cont(self, timesteps):
        target_percentile = timesteps * (self.max_percentile - self.min_percentile) + self.min_percentile
        return sigma_from_time_vectorized(target_percentile, self.continuous_noise, self.discrete_noise)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.continuous_noise = self.continuous_noise.to(*args, **kwargs)
        self.discrete_noise = self.discrete_noise.to(*args, **kwargs)
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
                     x0_tokens,
                     **kwargs):


        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0_tokens[:, :, None]
        ).squeeze(-1)
        nll = -1 * log_p_theta
        return nll

    def q_xt(self, x, alpha_t):
        t = (1 - alpha_t).squeeze()
        continuous_noise = self.get_continuous_from_discrete_noise(t)

        clean_embeddings = self.backbone.get_embedding(x)
        sigma = self.get_continuous_from_discrete_noise(t).squeeze()
        noise = torch.randn_like(clean_embeddings, device=x.device) * sigma[:, None, None]
        xt = clean_embeddings + noise

        return {
            'xt': xt,
            'discrete_noise': t,
            'continuous_noise': continuous_noise,
        }

    def prior_sample(self, *batch_dims):
        sigma_max = self.sigma_max
        noise = torch.randn(
            *batch_dims, self.backbone.config.hidden_size, dtype=torch.float32, device=self.device) * sigma_max
        return noise

    def forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs, reveal_mask=None, is_embed=True)
        return self._process_model_output(model_output=model_output)

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
        x = prompt_onehots * prompt_mask.unsqueeze(-1) + (1 - prompt_mask).unsqueeze(-1) * x
        return x, prompt_mask

    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)
        if prompt_tokens is not None and prompt_mask is not None:
            x, clean_mask = self.generate_sample_prompt(prompt_tokens, x, prompt_mask)
        timesteps = torch.linspace(.999, eps, num_steps + 1, device=self.device)
        continuous_noise = self.get_continuous_noise_sched_pure_cont(timesteps)

        self.max_sigma = continuous_noise.max().item()

        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i+1]

            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i+1]

            new_x, denoised = self._continuous_step(x, t, sigma_s=sigma_s, sigma_t=sigma_t, clean_mask=clean_mask, time_s=s, is_embed=self.is_embed)
            x = new_x

        final_tokens = sample_categorical(denoised)
        return final_tokens

    @torch.no_grad()
    def _continuous_step(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        clean_mask: torch.Tensor=None,
        is_embed=False,
    ) -> torch.Tensor:

        dt_cont_vec = torch.ones(x.shape[0], device=x.device) * (sigma_s - sigma_t).item()
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if clean_mask is None:
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)
        cond_denoised = self.forward(xt=x, discrete_noise=time_t_vec, reveal_mask=clean_mask, continuous_noise=sigma_t_vec, is_embed=is_embed).double()
        denoised = cond_denoised.exp()
        x0_hat = self.backbone.get_embedding(denoised)
        d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2)
        x_cont = x - dt_cont_vec[:, None, None]  * d
        return x_cont, denoised