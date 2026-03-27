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

class CANDI(Diffusion):
    def __init__(self, config, tokenizer):
        self.mask_index = len(tokenizer)
        vocab_size = len(tokenizer) + 1
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self._validate_configuration()
        self.save_hyperparameters()

        self.pure_continuous = config.algo.pure_continuous

        self.continuous_noise =torch.linspace(.2, 4.0, 1000)
        self.discrete_noise = expected_rank(d=self.vocab_size-1, a=torch.tensor(1).to(self.device), sigma=self.continuous_noise)

        self.continuous_noise = torch.tensor(self.continuous_noise)
        self.discrete_noise = torch.tensor(self.discrete_noise)

        self.step_size = config.algo.step_size

        self.min_percentile = config.algo.min_percentile
        self.max_percentile = config.algo.max_percentile

        self.sigma_min = config.algo.sigma_min
        self.sigma_max = config.algo.sigma_max
        self.temp = config.algo.temp
        self.sampler = config.algo.sampler
        self.is_embed = config.algo.is_embed
        self.use_percentile_scheduling = config.algo.use_percentile_scheduling

        self.cont_schedule_target = config.algo.cont_schedule_target
        self.cont_schedule_type = config.algo.cont_schedule_type



    def sample_t(self, n, accum_step):
        return super()._sample_t(n, accum_step)

    def _linear_sigma_schedule(self, discrete_noise):
        """Linear schedule directly in sigma space."""
        return self.sigma_min + discrete_noise * (self.sigma_max - self.sigma_min)
    
    
    def _cosine_sigma_schedule(self, discrete_noise):
        """Cosine schedule directly in sigma space."""
        cosine_interp = (1 - torch.cos(math.pi * discrete_noise)) / 2
        return self.sigma_min + cosine_interp * (self.sigma_max - self.sigma_min)
    
    
    def _geometric_sigma_schedule(self, discrete_noise):
        """Geometric/exponential schedule directly in sigma space (Song et al. VE SDE)."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** discrete_noise
    
    
    def _linear_r_schedule(self, discrete_noise):
        """Linear schedule in r*(t) space, inverted to sigma via equation (6)."""
        target_percentile = discrete_noise * (self.max_percentile - self.min_percentile) + self.min_percentile
        return sigma_from_time_vectorized(target_percentile, self.continuous_noise, self.discrete_noise)
    
    
    def _cosine_r_schedule(self, discrete_noise):
        """Cosine schedule in r*(t) space, inverted to sigma via equation (6)."""
        cosine_interp = (1 - torch.cos(math.pi * discrete_noise)) / 2
        target_percentile = cosine_interp * (self.max_percentile - self.min_percentile) + self.min_percentile
        return sigma_from_time_vectorized(target_percentile, self.continuous_noise, self.discrete_noise)
    
    
    def _geometric_r_schedule(self, discrete_noise):
        """Geometric schedule in r*(t) space, inverted to sigma via equation (6)."""
        target_percentile = self.min_percentile * (self.max_percentile / self.min_percentile) ** discrete_noise
        return sigma_from_time_vectorized(target_percentile, self.continuous_noise, self.discrete_noise)


    def get_continuous_from_discrete_noise(self, discrete_noise):
        if self.cont_schedule_target == 'sigma': 
            if self.cont_schedule_type == 'linear':
                return self._linear_sigma_schedule(discrete_noise)
            elif self.cont_schedule_type == 'cosine':
                return self._cosine_sigma_schedule(discrete_noise)
            elif self.cont_schedule_type == 'geometric':
                return self._geometric_sigma_schedule(discrete_noise)
            else:
                raise ValueError(f"Unknown cont_schedule_type: {self.cont_schedule_type}")
        elif self.cont_schedule_target == 'r':
            if self.cont_schedule_type == 'linear':
                return self._linear_r_schedule(discrete_noise)
            elif self.cont_schedule_type == 'cosine':
                return self._cosine_r_schedule(discrete_noise)
            elif self.cont_schedule_type == 'geometric':
                return self._geometric_r_schedule(discrete_noise)
            else:
                raise ValueError(f"Unknown cont_schedule_type: {self.cont_schedule_type}")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.continuous_noise = self.continuous_noise.to(*args, **kwargs)
        self.discrete_noise = self.discrete_noise.to(*args, **kwargs)
        return self

    def _validate_configuration(self):
        return

    def _process_model_output(self, xt, model_output, reveal_mask):
        if xt.ndim == 2:
            xt_tokens = xt
        else:
            xt_tokens = xt.argmax(dim=-1)
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
        if self.pure_continuous:
            nll = -1 * log_p_theta
        else:
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
        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)
        xt = torch.where(move_indices, uniform_tensor, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]

        return xt

    def q_xt(self, x, alpha_t):

        disc_xt = self.discrete_noising(x, alpha_t)
        reveal_mask = (disc_xt == x).float()

        discrete_noise = (1 - alpha_t).squeeze()
        continuous_noise = self.get_continuous_from_discrete_noise(discrete_noise)

        # accounting for the mask token
        onehot = F.one_hot(x, num_classes=self.vocab_size-1).float()


        xt_cont = onehot + continuous_noise[:, None, None].to(x.device) * torch.randn_like(onehot, device=x.device)

        xt = onehot * reveal_mask.unsqueeze(-1) + (1 - reveal_mask).unsqueeze(-1) * xt_cont
        return {
            'xt': xt,
            'reveal_mask': reveal_mask,
            'discrete_noise': discrete_noise,
            'continuous_noise': continuous_noise
        }

    def prior_sample(self, *batch_dims):
        sigma = self.get_continuous_from_discrete_noise(torch.tensor(.999).to(self.device))
        noise = torch.randn(
            *batch_dims, self.vocab_size-1, dtype=torch.float32, device=self.device
        )  * sigma
        return noise

    def forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs)
        return self._process_model_output(model_output=model_output, xt=kwargs['xt'], reveal_mask=kwargs['reveal_mask'])

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

    def generate_samples(self, **kwargs):
        if self.sampler == 'cached':
            return self.generate_samples_cache(**kwargs)
        else:
            return self.generate_samples_nocache(**kwargs)

    @torch.no_grad()
    def generate_samples_nocache(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

        if num_steps is None:
            num_steps = self.config.sampling.steps

        x = self.prior_sample(num_samples, self.num_tokens)


        clean_mask = torch.zeros((num_samples, self.num_tokens), device=x.device)

        if prompt_tokens is not None and prompt_mask is not None:
            x, clean_mask = self.generate_sample_prompt(prompt_tokens, x, prompt_mask)

        timesteps = torch.linspace(.999, eps, num_steps + 1, device=self.device)
        if self.use_percentile_scheduling:
            continuous_noise = self.get_continuous_from_discrete_noise(timesteps)
        else:
            continuous_noise = inference_sigmas(num_steps+1, self.sigma_min, self.sigma_max)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()

        self.prev_px0 = None
        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i+1]

            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i+1]

            x_cont, p_x0 = self._continuous_step(x, t, sigma_s=sigma_s, sigma_t=sigma_t, clean_mask=clean_mask, time_s=s)

            x, clean_mask = self._discrete_step(x_cont,
                                                p_x0, t, dt, prev_clean_mask=clean_mask)

        final_tokens = x.argmax(dim=-1)
        return final_tokens

    # optimized function with cached embeddings
    @torch.no_grad()
    def generate_samples_cache(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

        x = self.prior_sample(num_samples, self.num_tokens)
        if num_steps is None:
            num_steps = self.config.sampling.steps

        embedding_cache = self.backbone.get_embedding(x)
        clean_mask = torch.zeros((num_samples, self.num_tokens), device=x.device, dtype=torch.bool)
        if prompt_tokens is not None and prompt_mask is not None:
            x, clean_mask = self.generate_sample_prompt(prompt_tokens, x, prompt_mask)
        timesteps = torch.linspace(.999, eps, num_steps + 1, device=self.device)
        if self.use_percentile_scheduling:
            continuous_noise = self.get_continuous_from_discrete_noise(timesteps)
        else:
            continuous_noise = inference_sigmas(num_steps+1, self.sigma_min, self.sigma_max)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()


        # convert to tokens
        x = x.argmax(dim=-1)
        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i+1]

            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i+1]

            embedding_cache, x0_hat = self._continuous_step_cache(x, t,
                                                                sigma_s=sigma_s,
                                                                sigma_t=sigma_t,
                                                                clean_mask=clean_mask.float(),
                                                                time_s=s,
                                                                embedding_cache=embedding_cache,
                                                                is_first=(i==0))

            x, clean_mask = self._discrete_step_optimized(x0_hat, x, t, dt,
                                                        prev_clean_mask=clean_mask,
                                                    noise_removal_step=False)
        return x

    def _discrete_step_optimized(self, x0_hat, xt, t, dt, prev_clean_mask, noise_removal_step=False):

        if noise_removal_step:
            s=0
        else:
            s = t - dt

        # unmasking correspends to 1-alpha(s) / 1-alpha(t)
        # this is just s / t under a log linear schedule
        unmask = torch.rand(prev_clean_mask.shape, device=prev_clean_mask.device) < (t-s)/t
        xt[~prev_clean_mask] = x0_hat[~prev_clean_mask]
        new_clean_mask = prev_clean_mask | unmask
        return xt, new_clean_mask

    def _discrete_step(self, x_sigma, p_x0, t, dt, prev_clean_mask, noise_removal_step=False):
        if noise_removal_step:
            s = 0.0
        else:
            s = t - dt


        t_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * t.item()
        s_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * s.item()
        mask_probs = torch.ones((x_sigma.shape[0], x_sigma.shape[1], 1), device=x_sigma.device) * (s)
        unmasked_probs = p_x0 * (t_vec - s_vec)[:, None, None]

        q_xs = torch.cat([unmasked_probs, mask_probs], dim=-1)
        _x = sample_categorical(q_xs)
        new_clean_mask = (prev_clean_mask.bool() | (_x != self.mask_index)).float()


        # For tokens that got sampled to real values (not mask), use those
        old_x_tokens=x_sigma.argmax(dim=-1)


        # For tokens that got sampled to mask, keep old tokens but mark as not clean
        sampled_real_tokens = torch.where(_x != self.mask_index, _x, old_x_tokens)
        # Apply copy logic: keep old tokens where prev_clean_mask is True
        updated_tokens = torch.where(prev_clean_mask.bool(), old_x_tokens, sampled_real_tokens)
        updated_x = torch.nn.functional.one_hot(updated_tokens, num_classes=x_sigma.shape[-1]).float().to(x_sigma.device)

        updated_x = updated_x * new_clean_mask.unsqueeze(-1) + (1 - new_clean_mask).unsqueeze(-1) * x_sigma

        return updated_x, new_clean_mask

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
        if self.is_embed:
            x0_hat = self.backbone.get_embedding(denoised)
        else:
            x0_hat = denoised
        d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2)
        x_cont = x - dt_cont_vec[:, None, None]  * d
        return x_cont, denoised


    def _continuous_step_cache(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        embedding_cache: torch.Tensor,
        clean_mask: torch.Tensor=None,
        is_first=False,
    ) -> torch.Tensor:
        dt = sigma_s - sigma_t
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if clean_mask is None:
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)
        cond_denoised = self.forward(xt=x,
                                    discrete_noise=time_t_vec,
                                    reveal_mask=clean_mask,
                                    continuous_noise=sigma_t_vec,
                                    embedding=embedding_cache).double()


        denoised = cond_denoised.exp()
        x0_hat = sample_categorical(denoised)
        embedding_hat = self.backbone.get_embedding(x0_hat)
        d = (embedding_cache - embedding_hat) / (sigma_t ** 2)
        new_embedding_cache = embedding_cache - dt* d * self.step_size
        return new_embedding_cache, x0_hat
    
