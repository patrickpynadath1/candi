
from .absorbing_state import AbsorbingState, sample_categorical
from .registry import algo_registry
import pdb
import torch
from dataclasses import dataclass
import copy
from .encoder_decoder import EncoderDecoderPlain
from utils import get_aux_model

@dataclass
class Loss:
    loss: torch.FloatTensor
    latent_reconstruction: torch.FloatTensor
    num_tokens: torch.FloatTensor


@algo_registry.register("ldm")
class LDM(AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.latent_mode = config.algo.latent_mode
        self._initialize_encoder_decoder()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == "ancestral_cache" or self.sampler == "posterior"

    def _initialize_encoder_decoder(self):
        self.encoder_decoder = get_aux_model(self.config.algo.encoder_decoder_path,
                                             algo_registry=algo_registry, 
                                             tokenizer=self.tokenizer, 
                                             with_grads=False)
        return

    def _get_latent(self, x0, valid_tokens):
        with torch.no_grad():
            if self.latent_mode == "encoder_decoder":
                latent = self.encoder_decoder.backbone.encoder(input_tokens=x0, 
                                                            sigma=None, 
                                                            generation_mode=False)
            else: 
                latent = self.encoder_decoder.backbone(x0, 
                                                    sigma=None, 
                                                    get_logits=False)
        return latent
    
    def _decode_latent(self, z): 
        with torch.no_grad():
            if self.latent_mode == "encoder_decoder":
                logp_x0 = self.encoder_decoder.backbone.decode(input_latent=z, sigma=None)
            else: 
                logp_x0 = self.encoder_decoder.backbone.get_logits(z, sigma=None)
        return logp_x0

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
    
    def q_xt(self, x, alpha_t): 

        noise = torch.randn_like(x)
        x_t = x * alpha_t[:, None] + noise * (1 - alpha_t)[:, None]
        return x_t, noise
    

    # This is where the main training step goes 
    def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        (x0, output_tokens, valid_tokens) = self._process_model_input(
            x0, valid_tokens
        )
        if self.encoder_decoder is None:
            raise ValueError("Encoder-decoder model not initialized.")
        elif self.encoder_decoder.device != self.device:
            self.encoder_decoder.to(self.device)

        t = self._sample_t(x0.shape[0], current_accumulation_step)
        z_0 = self._get_latent(x0, valid_tokens).detach()

        assert t.shape[0] == x0.shape[0]
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        z_t, z_1 = self.q_xt(z_0, alpha_t)
        pred_v = self.backbone(input_latent = z_t, 
                               sigma=sigma.squeeze(), 
                               generation_mode=True)
        target_v = z_0 - z_1 
        latent_reconstruction_loss = torch.mean((pred_v - target_v) ** 2)

        return Loss(
            loss=latent_reconstruction_loss,
            latent_reconstruction=latent_reconstruction_loss,
            num_tokens=valid_tokens.sum(),
        )
    
    def _sample_latent_prior(self, batch_size):
        hidden_dim = self.config.model.hidden_size
        return torch.randn(batch_size, 128, hidden_dim, device=self.device)
    
    def _latent_fm_update(self, z, t, dt, d_cond=None, guidance_scale=2.0):
        _, alpha_t = self.noise(t)

        b_t = self.backbone(input_latent=z, sigma=self._sigma_from_alphat(alpha_t).squeeze(), input_tokens=None)
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
        self.metrics.update_train(losses.latent_reconstruction, losses.latent_reconstruction, losses.num_tokens)
        self.log(
            name="trainer/loss",
            value=losses.loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return losses.loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch["input_ids"], batch["attention_mask"])
        self.metrics.update_valid(losses.latent_reconstruction, losses.latent_reconstruction, losses.num_tokens)
        return losses.loss 
   
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):
        self._initialize_encoder_decoder()

        if num_steps is None:
            num_steps = self.config.sampling.steps 

        t = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps

        z = self._sample_latent_prior(num_samples)
        for i in range(num_steps): 
            t_i = t[i] * torch.ones(num_samples, 1, device=self.device)

            z = self._latent_fm_update(z, t_i, dt, d_cond=None)
        _, alpha_final = self.noise(t[-1].unsqueeze(0))

        logp_x0 = self._decode_latent(z)

        return logp_x0.argmax(dim=-1)
