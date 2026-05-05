from .diffusion import Diffusion
from .algo_utils import (
    expected_rank, 
    sigma_from_time_vectorized, 
    inference_sigmas, 
    sample_categorical, 
    training_sigma_ve
    )
from .registry import algo_registry
from .candi import CANDI
from .neighbor_blurring import *
import utils
import torch
import torch.nn.functional as F
import math

@algo_registry.register("candi_ra_change")
class CANDI_RA_CHANGE(CANDI):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        # set up conv function here 
        self.spatial_blurring_sigma = config.algo.spatial_blurring.get("sigma", 0.5)
        self.include_neighbors = config.algo.spatial_blurring.get("include_neighbors", True)
        self._precompute_kernel(self.device)
        self.temp = 1.0

    def _precompute_kernel(self, device): 
        # Precompute the kernel for neighbor blurring.
        seq_len = self.config.model.length
        self.kernel = gaussian_position_kernel(
            seq_len=seq_len,
            sigma=self.spatial_blurring_sigma,
            include_self=self.include_neighbors,
            device=device,
            dtype=torch.float32,
        )

    def to(self, *args, **kwargs):
        self.kernel.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    # TODO: modify this to incorporate a convolution at the sequence level before noising 
    def q_xt(self, x, alpha_t):

        disc_xt = self.discrete_noising(x, alpha_t)
        reveal_mask = (disc_xt == x).float()

        discrete_noise = (1 - alpha_t).squeeze()
        continuous_noise = self.get_continuous_from_discrete_noise(discrete_noise)

        # accounting for the mask token
        onehot = F.one_hot(x, num_classes=self.vocab_size-1).float()

        # spatially blurred means 
        mu_bar = neighbor_blurred_means(onehot, kernel=self.kernel)


        xt_cont = mu_bar + continuous_noise[:, None, None].to(x.device) * torch.randn_like(onehot, device=x.device)

        xt = onehot * reveal_mask.unsqueeze(-1) + (1 - reveal_mask).unsqueeze(-1) * xt_cont
        return {
            'xt': xt,
            'reveal_mask': reveal_mask,
            'discrete_noise': discrete_noise,
            'continuous_noise': continuous_noise
        }

    def generate_samples(self, **kwargs):
        return self.generation_loop(**kwargs)

    @torch.no_grad()
    def generation_loop(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):
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

        # compute the mixture 
        x0_hat = neighbor_blurred_means(denoised, kernel=self.kernel)
        d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2)
        x_cont = x - dt_cont_vec[:, None, None]  * d
        return x_cont, denoised