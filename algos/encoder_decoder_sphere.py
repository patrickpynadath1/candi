import pdb

from .diffusion import Diffusion
import torch
import torch.nn.functional as F
import math


class EncoderDecoderSphere(Diffusion):
    def __init__(self, config, tokenizer):
        self.mask_index = len(tokenizer)
        vocab_size = len(tokenizer)
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self._validate_configuration()
        self.save_hyperparameters()
        self.max_noise_level = math.tan(math.radians(config.algo.max_noise_angle))


    def sample_t(self, n, accum_step):
        return super()._sample_t(n, accum_step)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def _validate_configuration(self):
        return

    def _process_model_output(self, model_output):
        model_output = model_output / self.temp
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        return model_output

    def q_xt(self, x_clean, t):

        noise_level = t * self.max_noise_level
        sub_multiplier = torch.rand_like(t) * .5

        clean_latent = self.backbone.encode(x_clean, t)
        fixed_noise_vec = torch.randn_like(clean_latent)

        small_noise_latent = clean_latent + fixed_noise_vec * (sub_multiplier * noise_level)[:, None, None]
        large_noise_latent = clean_latent + fixed_noise_vec * noise_level[:, None, None]

        return {
            'small_noise_latent': small_noise_latent,
            'large_noise_latent': large_noise_latent,
            'clean_latent': clean_latent,
            'noise_level': noise_level,
        }

    def prior_sample(self, *batch_dims):
        x = torch.randn(*batch_dims, self.config.model.n_latents, self.config.model.hidden_size, device=self.device)
        x = self.backbone.spherify(x)
        return x

    def forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs)
        return self._process_model_output(model_output=model_output)

    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches

        x0 = batch['input_ids']
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]

        
        noisy_outputs = self.q_xt(x0, t)

        small_noise_latent = noisy_outputs['small_noise_latent']
        large_noise_latent = noisy_outputs['large_noise_latent']
        clean_latent = noisy_outputs['clean_latent']

        # small noise nll
        pred_x0 = self.backbone.decode(small_noise_latent, t, spherify_first=True).log_softmax(dim=-1)
        reconstruction_nll = pred_x0.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1) * -1

        # cos sim
        decoded_hidden_reps = self.backbone.decode(small_noise_latent, t, spherify_first=True, skip_embedding=True)
        recoded_latent = self.backbone.encode(decoded_hidden_reps, t, skip_embedding=True)
        cos_sim = F.cosine_similarity(clean_latent, recoded_latent, dim=-1)

        # kl divergence between large noise and small noise
        pred_x0_noisy = self.backbone.decode(large_noise_latent, t, spherify_first=False).log_softmax(dim=-1)
        kl_noisy = F.kl_div(pred_x0_noisy, pred_x0.detach(), reduction='none', log_target=True).sum(dim=-1)

        self.log('train/reconstruction_nll', reconstruction_nll.mean().item(), on_step=True, prog_bar=True, logger=True)
        self.log('train/kl_noisy', kl_noisy.mean().item(), on_step=True, prog_bar=True, logger=True)
        self.log('train/cos_sim', cos_sim.mean().item(), on_step=True, prog_bar=True, logger=True)
        self.log('train/decoded_hidden_rep_norm', decoded_hidden_reps.norm(dim=-1).mean().item(), on_step=True, prog_bar=True, logger=True)
        total_loss = (reconstruction_nll.mean(dim=-1) + kl_noisy.mean(dim=-1) + cos_sim.mean(dim=-1)).mean()

        self.log('train/total_loss', total_loss.item(), on_step=True, prog_bar=True, logger=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches

        x0 = batch['input_ids']
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]

        
        noisy_outputs = self.q_xt(x0, t)

        small_noise_latent = noisy_outputs['small_noise_latent']
        large_noise_latent = noisy_outputs['large_noise_latent']
        clean_latent = noisy_outputs['clean_latent']

        # small noise nll
        pred_x0 = self.backbone.decode(small_noise_latent, t, spherify_first=True).log_softmax(dim=-1)
        reconstruction_nll = pred_x0.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1) * -1

        # cos sim
        decoded_hidden_reps = self.backbone.decode(small_noise_latent, t, spherify_first=True, skip_embedding=True)
        recoded_latent = self.backbone.encode(decoded_hidden_reps, t, skip_embedding=True)
        cos_sim = F.cosine_similarity(clean_latent, recoded_latent, dim=-1)

        # kl divergence between large noise and small noise
        pred_x0_noisy = self.backbone.decode(large_noise_latent, t, spherify_first=False).log_softmax(dim=-1)
        kl_noisy = F.kl_div(pred_x0_noisy, pred_x0.detach(), reduction='none', log_target=True).sum(dim=-1)

        self.log('val/reconstruction_nll', reconstruction_nll.mean().item(), on_step=True, prog_bar=True, logger=True)
        self.log('val/kl_noisy', kl_noisy.mean().item(), on_step=True, prog_bar=True, logger=True)
        self.log('val/cos_sim', cos_sim.mean().item(), on_step=True, prog_bar=True, logger=True)
        self.log('val/decoded_hidden_rep_norm', decoded_hidden_reps.norm(dim=-1).mean().item(), on_step=True, prog_bar=True, logger=True)
        total_loss = (reconstruction_nll.mean(dim=-1)).mean()

        self.log('val/total_loss', total_loss.item(), on_step=True, prog_bar=True, logger=True) 
        return total_loss

    @torch.no_grad()
    def generate_sample(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

        # 1) sample Gaussian prior
        x_noisy = self.prior_sample(num_samples, self.num_tokens)

        if num_steps is None:
            num_steps = self.config.sampling.steps

        # 2) spherify initial sample
        x_noisy = self.spherify(x_noisy)

        # 3) use fixed σ_max during sampling
        sigmas = torch.ones(num_samples, device=self.device) * self.max_noise_level

        # 4) sample ONE noise direction (shared across steps)
        noise = torch.randn_like(x_noisy)

        for i in range(num_steps):

            # ---- Decode ----
            output_hidden_rep = self.backbone.decode(
                x_noisy,
                sigma=sigmas,
                spherify_first=False,
                skip_embedding=True
            )

            # ---- Re-encode ----
            v_clean = self.backbone.encode(
                output_hidden_rep,
                sigmas,
                skip_embedding=True
            )

            # ---- Add angular noise ----
            v_noisy = v_clean + sigmas[:, None] * noise

            # ---- Re-project to sphere ----
            x_noisy = self.spherify(v_noisy)

        # final decode
        x = self.backbone.decode(
            x_noisy,
            sigma=sigmas,
            spherify_first=False,
            skip_embedding=False
        )

        x_tok = x.argmax(dim=-1)
        return x_tok
