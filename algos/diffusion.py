import utils
from .trainer_base import TrainerBase


import torch

from samplers import SamplingRequest, create_sampler, run_sampler


class Diffusion(TrainerBase):
    sampling_family = "discrete_diffusion"
    sampling_capabilities = frozenset({
        "discrete_prior",
        "noise_schedule",
        "greedy_denoise",
    })

    def __init__(self, config, tokenizer, vocab_size=None):
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.paper_time_sampler = None
        time_config = config.get("time_sampling", {})
        if (
            time_config.get("adaptive", False)
            or config.algo.get("paper_time_conditioning", False)
            or config.sampling.predictor == "paper_discrete"
            or self.sampling_family == "simplex"
        ):
            from .paper_time_sampler import PaperTimeSampler

            self.paper_time_sampler = PaperTimeSampler(**dict(time_config))

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.noise_removal in {"none", "ancestral", "greedy"}
        assert self.loss_type in {"elbo", "low_var"}
        if self.config.sampling.noise_removal == "greedy":
            assert self.config.sampling.predictor != "analytic"
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
        if getattr(getattr(self, "paper_time_sampler", None), "adaptive", False):
            t = self.paper_time_sampler(t)
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t

    def _sigma_from_alphat(self, alpha_t):
        if getattr(getattr(getattr(self, "config", None), "algo", None), "paper_time_conditioning", False):
            return 1 - alpha_t
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

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False,
            valid_tokens=None):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        dalpha_t, alpha_t = self.noise(t)
        dalpha_t = dalpha_t.unsqueeze(-1)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)

        log_x_theta = self.forward(xt, sigma=sigma)
        utils.print_nans(log_x_theta, "model_output")
        loss = self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == "low_var",
        )
        if self.loss_type == "ce":
            loss = -log_x_theta.gather(-1, x0[..., None]).squeeze(-1)
        self._record_time_loss(t, loss, valid_tokens, train_mode)
        if train_mode and getattr(getattr(self, "paper_time_sampler", None), "adaptive", False):
            ce = -log_x_theta.gather(-1, x0[..., None]).squeeze(-1)
            self.paper_time_sampler.observe(t, ce, valid_tokens, self.global_step)
        return loss

    def _get_score(self, **kwargs):
        del kwargs
        raise NotImplementedError

    def _denoiser_update(self, x, t):
        raise NotImplementedError

    def _analytic_update(self, x, t, dt):
        raise NotImplementedError

    def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
        raise NotImplementedError

    def sample_ancestral_transition(self, **kwargs):
        return self._ancestral_update(**kwargs)

    def sample_analytic_transition(self, **kwargs):
        return self._analytic_update(**kwargs)

    def sample_denoiser_transition(self, **kwargs):
        return self._denoiser_update(**kwargs)

    def sample_greedy_denoise(self, x, t):
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        return self.forward(xt=x, sigma=sigma).argmax(dim=-1)

    def sample_denoising_log_probs(self, x, t):
        """Return normalized clean-token predictions for a noisy state."""
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        return self.forward(xt=x, sigma=sigma)

    def sample_zero_time_denoise(self, x, ones):
        return self.forward(x, 0 * ones).argmax(dim=-1)

    def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        # TODO(subham): Test this method after refactoring.
        self._eval_mode()
        try:
            sampler = create_sampler(self.config, default="semi_ar")
            result = run_sampler(
                sampler,
                self,
                SamplingRequest(
                    num_samples=self.config.loader.eval_batch_size,
                    options={
                        "stride_length": stride_length,
                        "num_strides": num_strides,
                        "dt": dt,
                    },
                ),
            )
            chunks = result.metadata["chunks"]
            sequence_lengths = (
                (result.samples[:, 1:] == self.tokenizer.eos_token_id)
                .cumsum(-1)
                .eq(0)
                .sum(-1)
                .cpu()
                .numpy()
            )
            intermediate_text_samples = [
                self.tokenizer.batch_decode(torch.cat(chunks[:i], dim=1))
                for i in range(2, len(chunks) + 1)
            ]
            return result.metadata["nfe"], intermediate_text_samples, sequence_lengths
        finally:
            self._train_mode()
