import utils
from .trainer_base import TrainerBase


import numpy as np
import torch


class Diffusion(TrainerBase):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.noise_removal in {"none", "ancestral", "greedy"}
        assert self.loss_type in {"elbo", "low_var"}
        if self.config.sampling.noise_removal == "greedy":
            assert self.sampler != "analytic"
            assert self.parameterization in {"mean", "subs"}

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _process_sigma(self, sigma):
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n, accum_step):
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t

    def _sigma_from_alphat(self, alpha_t):
        return -torch.log(alpha_t)

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype, device=self.device)
        sigma_t0 = self._sigma_from_alphat(self.noise(t0)[1])
        model_output_t0 = self.forward(x0, sigma_t0)
        return -torch.gather(
            input=model_output_t0, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

    def nll_per_token(self, model_output, xt, x0, alpha_t, dalpha_t, low_var):
        raise NotImplementedError

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)

        log_x_theta = self.forward(xt, sigma=sigma)
        utils.print_nans(log_x_theta, "model_output")
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == "low_var",
        )

    def _get_score(self, **kwargs):
        del kwargs
        raise NotImplementedError

    def _denoiser_update(self, x, t):
        raise NotImplementedError

    def _analytic_update(self, x, t, dt):
        raise NotImplementedError

    def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
        raise NotImplementedError

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_mask=None, prompt_tokens=None):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)
        if prompt_tokens is not None and prompt_mask is not None:

            # assumes that they will be the same shape
            x = (prompt_mask * prompt_tokens + (1 - prompt_mask) * x).long()


        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ancestral":
                # for mdlm, because of SUBS, the prompt mask wont be used 
                # but is needed to make duo / usdms condition on promtps / infill

                _, x = self._ancestral_update(x=x, t=t, dt=dt, p_x0=None, prompt_mask=prompt_mask)
            elif self.sampler == "ancestral_cache":
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next


            else:
                x = self._analytic_update(x=x, t=t, dt=dt)

        t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        if self.config.sampling.noise_removal == "ancestral":
            if self.sampler == "analytic":
                x = self._denoiser_update(x=x, t=t0)
            else:
                _, x = self._ancestral_update(
                    x=x, t=t0, dt=None, p_x0=p_x0_cache, noise_removal_step=True, prompt_mask=prompt_mask
                )
        elif self.config.sampling.noise_removal == "greedy":
            sigma = self._sigma_from_alphat(self.noise(t0)[1])
            x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)

        if self.token_offset:
            x += self.tokenizer.eos_token_id
        return x

    @torch.no_grad
    def _semi_ar_sampler(self, n_samples, stride_length, num_strides, dt=0.001):
        # TODO(subham): Test this method after refactoring.
        ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self.prior_sample(n_samples, self.num_tokens)
            if target is not None:
                x[:, :-stride_length] = target
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]

        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = (
            (
                np.concatenate(intermediate_tokens, axis=1)[:, 1:]
                == self.tokenizer.eos_token_id
            ).cumsum(-1)
            == 0
        ).sum(-1)
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)
                )
            )
        return (sampling_steps, intermediate_text_samples, sequence_lengths)

    def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        # TODO(subham): Test this method after refactoring.
        self._eval_mode()
        try:
            return self._semi_ar_sampler(
                n_samples=self.config.loader.eval_batch_size,
                stride_length=stride_length,
                num_strides=num_strides,
                dt=dt,
            )
        finally:
            self._train_mode()
