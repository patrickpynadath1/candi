
from .absorbing_state import AbsorbingState, sample_categorical
from .registry import algo_registry
import pdb
import torch
from dataclasses import dataclass
import copy

@dataclass
class Loss:
    loss: torch.FloatTensor
    token_reconstruction_loss: torch.FloatTensor
    latent_reconstruction_loss: torch.FloatTensor
    denoised_token_reconstruction_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor


@algo_registry.register("h_candi_v1")
class H_CANDI_v1(AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == "ancestral_cache" or self.sampler == "posterior"

    def _initialize_ema_encoder(self): 

        ema_weight = .999
        self.encoder_ema = copy.deepcopy(self.backbone.encoder)

        pass 


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
    
    # helper function for getting nlls 
    def _get_nll(self, raw_logits, x0, valid_tokens, weights = None):
        logits = raw_logits.log_softmax(dim=-1)
        log_p_theta = torch.gather(
            input=logits, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)
        token_reconstruction_loss = log_p_theta * -1
        if weights is not None:
            tr_loss_mean = (token_reconstruction_loss * valid_tokens * weights[:, None]).sum() / (valid_tokens * weights[:, None]).sum()
        else:
            tr_loss_mean = (token_reconstruction_loss * valid_tokens).sum() / valid_tokens.sum()
        return tr_loss_mean


    # This is where the main training step goes 
    def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        (x0, output_tokens, valid_tokens) = self._process_model_input(
            x0, valid_tokens
        )
        t = self._sample_t(x0.shape[0], current_accumulation_step) * .3
        assert t.shape[0] == x0.shape[0]

        if self.global_step < 500:
            latent_coeff = 0
        else: 
            latent_coeff = min(1, (self.global_step  - 500)/ 500)
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)

        # first loss: logit reconstruction
        clean_sigma = torch.ones(x0.shape[0], device=self.device) * .001
        latents = self.backbone.encoder(input_tokens=xt, sigma=clean_sigma, generation_mode=False)
        latents_for_flow = latents.detach()
        small_peturb = latents + .7 * torch.randn_like(latents)
        logits_raw = self.backbone.decode(input_latent=small_peturb, sigma=clean_sigma)

        # computing log p_theta(x0)
        tr_loss_mean = self._get_nll(logits_raw, x0, valid_tokens)

        # second loss: latent reconstruction
        noise = torch.randn_like(latents)
        t_2 = self._sample_t(x0.shape[0], current_accumulation_step)
        _, alpha_t_2 = self.noise(t_2)
        sigma_t_2 = self._sigma_from_alphat(alpha_t_2.unsqueeze(-1))
        corrupted_latents = latents_for_flow * (1 - t_2[:, None, None]) + noise * t_2[:, None, None]
        denoised_latents = self.backbone.denoiser(input_latent = corrupted_latents, sigma=sigma_t_2.squeeze(), generation_mode=True)

        latent_target = latents_for_flow - noise
        latent_reconstruction_loss = torch.mean((denoised_latents - latent_target) ** 2)
        # latent_tr_loss_mean = self._get_nll(logits_from_corr, x0, valid_tokens, weights = (1 - t_2))

        return Loss(
            loss=tr_loss_mean + latent_reconstruction_loss * latent_coeff,
            token_reconstruction_loss=tr_loss_mean,
            latent_reconstruction_loss=latent_reconstruction_loss,
            num_tokens=valid_tokens.sum(),
            denoised_token_reconstruction_loss=tr_loss_mean,  # latent_tr_loss_mean
        )

    def q_xt(self, x, alpha_t):
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

    def on_after_backward(self):
        ge_grad_norm = torch.stack([
            p.grad.norm() for p in self.backbone.encoder.parameters() 
            if p.grad is not None
        ]).mean()
        
        decoder_grad_norm = torch.stack([
            p.grad.norm() for p in self.backbone.decoder.parameters() 
            if p.grad is not None
        ]).mean()
        
        self.log("grad_norm/encoder", ge_grad_norm)
        self.log("grad_norm/decoder", decoder_grad_norm)

    
    def _sample_latent_prior(self, batch_size):
        hidden_dim = self.config.model.hidden_size
        return torch.randn(batch_size, self.config.model.k_register_latents, hidden_dim, device=self.device)
    
    def _latent_fm_update(self, z, t, dt, d_cond=None, guidance_scale=2.0):
        _, alpha_t = self.noise(t)

        d_uncond = self.backbone.denoise(input_latents=z, sigma=self._sigma_from_alphat(alpha_t).squeeze(), input_tokens=None)

        if d_cond is not None:
            d = d_uncond + guidance_scale * (d_cond - d_uncond)
        else: 
            d = d_uncond
        # small numerical for stability
        b_t = (d - z) / (t.unsqueeze(-1) + 1e-5)
        new_z = z + b_t * dt
        return new_z

    def training_step(self, batch, batch_idx):

        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            0,
            train_mode=True,
        )
        self.metrics.update_train(losses.token_reconstruction_loss, losses.latent_reconstruction_loss, losses.num_tokens)
        self.log(
            name="trainer/loss",
            value=losses.loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            name="trainer/token_reconstruction_loss",
            value=losses.token_reconstruction_loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            name="trainer/denoised_token_reconstruction_loss",
            value=losses.denoised_token_reconstruction_loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            name="trainer/latent_reconstruction_loss",
            value=losses.latent_reconstruction_loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return losses.loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch["input_ids"], batch["attention_mask"])
        self.metrics.update_valid(losses.token_reconstruction_loss, losses.latent_reconstruction_loss, losses.num_tokens)
        return losses.loss 
   
    # @torch.no_grad()
    # def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

    #     if num_steps is None:
    #         num_steps = self.config.sampling.steps 

    #     t = torch.linspace(1, eps, num_steps + 1, device=self.device)
    #     dt = (1 - eps) / num_steps

    #     z = self._sample_latent_prior(num_samples)
    #     for i in range(num_steps): 
    #         t_i = t[i] * torch.ones(num_samples, 1, device=self.device)

    #         z = self._latent_fm_update(z, t_i, dt, d_cond=None)
    #     _, alpha_final = self.noise(t[-1].unsqueeze(0))

    #     sigma_final = self._sigma_from_alphat(alpha_final).squeeze() * torch.ones(num_samples, device=self.device)
    #     logp_x0 = self.backbone.decode(input_latent=z, sigma=sigma_final)

    #     return logp_x0.argmax(dim=-1)
 

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

        if num_steps is None:
            num_steps = self.config.sampling.steps 

        inner_loop_steps = 8
        outer_loop_steps = num_steps // inner_loop_steps

        t_outer = torch.linspace(1, eps, outer_loop_steps + 1, device=self.device)
        t_inner = torch.linspace(1, eps, inner_loop_steps + 1, device=self.device)

        dt_outer = (1 - eps) / outer_loop_steps
        dt_inner = (1 - eps) / inner_loop_steps

        z_cond = None
        xt = self.prior_sample(num_samples, self.config.model.length).long()
        for i in range(outer_loop_steps): 
            z_uncond = self._sample_latent_prior(num_samples)
            t_o = t_outer[i] * torch.ones(num_samples, 1, device=self.device)

            # denoise latent
            for j in range(inner_loop_steps):
                t_i = t_inner[j] * torch.ones(num_samples, 1, device=self.device)
                z_uncond = self._latent_fm_update(z_uncond, t_o, dt_inner, d_cond=z_cond)

            # decode latent into logits 
            _, alpha_t = self.noise(t_o)
            sigma = self._sigma_from_alphat(alpha_t)
            logp_x0 = self.backbone.decode(input_latent=z_uncond, sigma=sigma.squeeze())
            logp_x0 = self._process_model_output(logp_x0, xt, sigma.squeeze())
            
            is_final = i == outer_loop_steps - 1
            _, xt = self._ancestral_update(xt, 
                                           t_o, 
                                           dt_outer, 
                                           logp_x0, 
                                           noise_removal_step=is_final)
            
            if is_final: 
                break 
            else:
                t_next = t_outer[i + 1] * torch.ones(num_samples, 1, device=self.device)
                _, alpha_next = self.noise(t_next)
                sigma_next = self._sigma_from_alphat(alpha_next)
                z_cond = self.backbone.encode(input_tokens=xt, sigma=sigma_next.squeeze())
        return xt

    def _ancestral_update(self, x, t, dt, logp_x0, noise_removal_step=False, **kwargs):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2
        p_x0 = logp_x0.exp()
        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        _x = sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def prior_sample(self, *batch_dims):
        # pdb.set_trace()
        return self.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.device
        )

    
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