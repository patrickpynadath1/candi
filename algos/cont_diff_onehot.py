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

@algo_registry.register("cont_diff_onehot")
class ContDiffOnehot(Diffusion):
    def __init__(self, config, tokenizer):
        self.mask_index = len(tokenizer)
        vocab_size = len(tokenizer) + 1
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

    def sample_t(self, n, accum_step):
        return super()._sample_t(n, accum_step)

    def get_continuous_from_discrete_noise(self, discrete_noise):
        target_percentile = discrete_noise * (self.max_percentile - self.min_percentile) + self.min_percentile
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
                     alpha_t,
                     dalpha_t,
                     **kwargs):


        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0_tokens[:, :, None]
        ).squeeze(-1)
        nll = -1 * log_p_theta
        return nll

    def q_xt(self, x, alpha_t):

        t = (1 - alpha_t).squeeze()
        continuous_noise = self.get_continuous_from_discrete_noise(t)

        # accounting for the mask token
        onehot = F.one_hot(x, num_classes=self.vocab_size).float()
        sigma = continuous_noise
        noise = torch.randn_like(onehot, device=x.device) * sigma[:, None, None]
        xt = onehot + noise
        return {
            'xt': xt,
            'discrete_noise': t,
            'continuous_noise': continuous_noise,
            'is_embed': False
        }

    def prior_sample(self, *batch_dims):

        sigma = self.get_continuous_from_discrete_noise(torch.tensor(.999).to(self.device))
        noise = torch.randn(
            *batch_dims, self.vocab_size, dtype=torch.float32, device=self.device
        )  * sigma
        return noise

    def forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs, reveal_mask=None)
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


    def generate_samples_pure_cont(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)
        clean_mask = torch.zeros((num_samples, self.num_tokens), device=x.device)
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

            new_x, denoised = self._continuous_step(x, t, sigma_s=sigma_s, sigma_t=sigma_t, time_s=s)
            x = new_x

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
    ) -> torch.Tensor:

        dt_cont_vec = torch.ones(x.shape[0], device=x.device) * (sigma_s - sigma_t).item()
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        cond_denoised = self.forward(xt=x, discrete_noise=time_t_vec, continuous_noise=sigma_t_vec).double()
        denoised = cond_denoised.exp()
        if self.is_embed:
            x0_hat = self.backbone.get_embedding(denoised)
        else:
            x0_hat = denoised
        d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2)
        x_cont = x - dt_cont_vec[:, None, None]  * d
        return x_cont, denoised
