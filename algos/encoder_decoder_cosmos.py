
from .absorbing_state import AbsorbingState, sample_categorical
from .registry import algo_registry
import pdb
import torch
from dataclasses import dataclass
import copy
from utils import get_aux_model

@dataclass
class Loss:
    loss: torch.FloatTensor
    token_reconstruction_loss: torch.FloatTensor
    mse_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor


# replication of encoder-decoder from https://arxiv.org/pdf/2506.21170
@algo_registry.register("encoder_decoder_cosmos")
class EncoderDecoderCosmos(AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.hr_vector_smoothing = config.algo.hr_vector_smoothing
        self.hr_vector_dropout = config.algo.hr_vector_dropout
        self.latent_feature_dropout = config.algo.latent_feature_dropout

        # loading mlm 
        self._load_mlm()

    def _load_mlm(self): 
        self.mlm = get_aux_model(self.config.algo.masked_denoiser_path, 
                                algo_registry=algo_registry,
                                tokenizer=self.tokenizer,
                                with_grads=False)
        return

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == "ancestral_cache" or self.sampler == "posterior"

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
    
    def _get_h0(self, x0): 
        return self.mlm.backbone(x0, sigma=None, get_logits=False)
    
    # mention on page 5 in latent space augmentations
    def _apply_latent_dropout(self, z0): 
        if self.latent_feature_dropout > 0:
            feature_mask = torch.rand(z0.shape, device=z0.device) > self.latent_feature_dropout
            z0 = z0 * feature_mask.float()
        return z0

    # formula from activate space perturbations, page 5 section 4.2.2 b 
    def _apply_hr_vector_smoothing(self, hr_vector):
        delta = self.hr_vector_smoothing 
        h_tilde = hr_vector * delta + ((1 - delta ** 2) ** .5) * torch.randn_like(hr_vector)
        return h_tilde
    
    # formula from activate space perturbations, page 5 section 4.2.2 b 
    def _apply_hr_vector_dropout(self, hr_vector): 
        dropout_mask = torch.rand(hr_vector.shape[:-1], device=hr_vector.device) > self.hr_vector_dropout
        hr_vector = hr_vector * dropout_mask.float()[..., None]
        return hr_vector
    
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

    def _get_mse(self, h_hat, h0, valid_tokens):
        mse_loss = ((h_hat - h0) ** 2) * valid_tokens[..., None]
        return mse_loss.sum() / (valid_tokens.sum() * h0.shape[-1])


    # This is where the main training step goes 
    def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        (x0, output_tokens, valid_tokens) = self._process_model_input(
            x0, valid_tokens
        )

        # getting smooth + dropped hr vector
        h0 = self._get_h0(x0)
        h0_smooth = self._apply_hr_vector_smoothing(h0)
        h0_smooth_dropout = self._apply_hr_vector_dropout(h0_smooth)

        # getting latent 
        z0 = self.backbone.encoder(h0_smooth_dropout, sigma=None, skip_embedding=True)
        z0 = self._apply_latent_dropout(z0)

        # getting logits + hidden representations 
        h_hat, logits = self.backbone.decoder(input_latent=z0, sigma=None, return_hr_vector=True)

        mse = self._get_mse(h_hat, h0, valid_tokens)
        ce = self._get_nll(logits, x0, valid_tokens)

        return Loss(
            loss=mse + ce,
            token_reconstruction_loss=ce,
            mse_loss=mse,
            num_tokens=valid_tokens.sum(),
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

    
    def training_step(self, batch, batch_idx):

        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            0,
            train_mode=True,
        )
        self.metrics.update_train(losses.loss, losses.token_reconstruction_loss, num_tokens=1)
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
        self.metrics.update_valid(losses.loss, losses.token_reconstruction_loss, num_tokens=1)
        return losses.loss 
   

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

        return None

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