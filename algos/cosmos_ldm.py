from dataclasses import dataclass

import torch

from .diffusion import Diffusion
from .cosmos_algo_utils.dynamic import DynamicSDE
from .cosmos_algo_utils.solvers import create_solver
from .registry import algo_registry
from utils import get_aux_model


@dataclass
class Loss:
    loss: torch.FloatTensor
    latent_reconstruction: torch.FloatTensor
    num_tokens: torch.FloatTensor


@algo_registry.register("cosmos_ldm")
class CosmosLDM(Diffusion):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

        self.encoder_decoder = None
        self.use_self_cond = config.algo.diffusion.use_self_cond

        self.register_buffer("latent_mean", torch.empty(0), persistent=True)
        self.register_buffer("latent_std", torch.empty(0), persistent=True)
        self.register_buffer("latent_stats_initialized", torch.tensor(False), persistent=True)

        self.dynamic = DynamicSDE(config)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=self.calc_score,
            ode_sampling=config.algo.dynamic.ode_sampling,
        )

        self._initialize_encoder_decoder()

    def _validate_configuration(self):
        return

    def _initialize_encoder_decoder(self):
        self.encoder_decoder = get_aux_model(
            self.config.algo.encoder_decoder_path,
            algo_registry=algo_registry,
            tokenizer=self.tokenizer,
            with_grads=False,
        )

    def _ensure_encoder_decoder_device(self):
        if self.encoder_decoder is None:
            raise ValueError("Encoder-decoder model not initialized.")
        if self.encoder_decoder.device != self.device:
            self.encoder_decoder.to(self.device)

    def _set_latent_stats(self, latent_mean, latent_std):
        latent_mean = latent_mean.detach().float()
        latent_std = latent_std.detach().float().clamp_min(1e-4)

        if self.latent_mean.numel() == 0:
            self.latent_mean = latent_mean
            self.latent_std = latent_std
        else:
            self.latent_mean.copy_(latent_mean)
            self.latent_std.copy_(latent_std)
        self.latent_stats_initialized.fill_(True)

    @property
    def has_latent_stats(self):
        return bool(self.latent_stats_initialized.item())

    @torch.no_grad()
    def _compute_latent_statistics(self):
        self._ensure_encoder_decoder_device()

        val_dataloaders = self.trainer.val_dataloaders
        if not isinstance(val_dataloaders, (list, tuple)):
            val_dataloaders = [val_dataloaders]
        if not val_dataloaders:
            raise ValueError("Validation dataloader is required to compute latent statistics.")

        was_training = self.training
        self.eval()
        self.encoder_decoder.eval()

        latent_sum = None
        latent_sum_sq = None
        latent_count = torch.zeros(1, device=self.device, dtype=torch.float32)

        for dataloader in val_dataloaders:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                latents = self._get_latent(input_ids, attention_mask).float()

                if latent_sum is None:
                    latent_sum = torch.zeros_like(latents[0], dtype=torch.float32)
                    latent_sum_sq = torch.zeros_like(latents[0], dtype=torch.float32)

                latent_sum += latents.sum(dim=0)
                latent_sum_sq += latents.pow(2).sum(dim=0)
                latent_count += latents.shape[0]

        if latent_sum is None or latent_count.item() == 0:
            raise ValueError("Could not compute latent statistics from validation data.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(latent_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(latent_sum_sq, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(latent_count, op=torch.distributed.ReduceOp.SUM)

        latent_mean = latent_sum / latent_count
        latent_var = (latent_sum_sq / latent_count) - latent_mean.pow(2)
        latent_std = latent_var.clamp_min(1e-4).sqrt()
        self._set_latent_stats(latent_mean, latent_std)

        if was_training:
            self.train()

    def normalize_latent(self, latent):
        if not self.has_latent_stats:
            return latent
        return (latent - self.latent_mean) / self.latent_std

    def denormalize_latent(self, latent):
        if not self.has_latent_stats:
            return latent
        return latent * self.latent_std + self.latent_mean

    def _get_latent(self, x0, valid_tokens):
        self._ensure_encoder_decoder_device()
        with torch.no_grad():
            h0 = self.encoder_decoder._get_h0(x0)
            h0 = self.encoder_decoder.backbone.enc_normalizer.normalize(h0)
            latent = self.encoder_decoder.backbone.encoder(
                token_ids=x0,
                mask_tokens=valid_tokens,
                token_embeddings=h0,
            )
        return latent

    def _decode_latent(self, z):
        self._ensure_encoder_decoder_device()
        with torch.no_grad():
            logits = self.encoder_decoder.backbone.decoder(
                encoder_latents=z,
                return_last_hidden_state=False,
            )
        return logits

    def _sample_time(self, batch_size):
        diffusion_cfg = self.config.algo.diffusion
        return torch.rand(batch_size, device=self.device) * (diffusion_cfg.T - diffusion_cfg.eps) + diffusion_cfg.eps

    def q_xt(self, x, alpha_t):
        del alpha_t
        raise NotImplementedError("CosmosLDM uses continuous latent noising, not token-space q_xt.")

    def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        del train_mode
        x0, _, valid_tokens = self._process_model_input(x0, valid_tokens)

        clean_latent = self._get_latent(x0, valid_tokens).detach()
        clean_latent = self.normalize_latent(clean_latent)

        t = self._sample_time(x0.shape[0])
        marginal_forward = self.dynamic.marginal(clean_latent, t)
        z_t = marginal_forward["x_t"]

        x_0_self_cond = torch.zeros_like(clean_latent)
        if self.use_self_cond and torch.rand(1, device=self.device).item() > 0.5:
            with torch.no_grad():
                x_0_self_cond = self.backbone(
                    x_t=z_t.clone(),
                    time_t=t.clone(),
                    x_0_self_cond=x_0_self_cond,
                ).detach()

        z_0_pred = self.backbone(
            x_t=z_t,
            time_t=t,
            x_0_self_cond=x_0_self_cond,
        )

        latent_reconstruction_loss = torch.mean((clean_latent - z_0_pred) ** 2)

        return Loss(
            loss=latent_reconstruction_loss,
            latent_reconstruction=latent_reconstruction_loss,
            num_tokens=valid_tokens.sum(),
        )

    def calc_score(self, x_t, t, x_0_self_cond):
        params = self.dynamic.marginal_params(t)
        x_0 = self.backbone(
            x_t=x_t,
            time_t=t,
            x_0_self_cond=x_0_self_cond,
        )
        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta,
        }

    def on_train_start(self):
        super().on_train_start()
        self._ensure_encoder_decoder_device()
        if not self.has_latent_stats:
            self._compute_latent_statistics()

    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            current_accumulation_step,
            train_mode=True,
        )
        self.metrics.update_train(
            losses.latent_reconstruction,
            losses.latent_reconstruction,
            losses.num_tokens,
        )
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
        self.metrics.update_valid(
            losses.latent_reconstruction,
            losses.latent_reconstruction,
            losses.num_tokens,
        )
        return losses.loss

    def _sample_latent_prior(self, batch_size):
        self._ensure_encoder_decoder_device()
        num_latents = self.encoder_decoder.backbone.encoder.num_latents
        latent_dim = self.encoder_decoder.backbone.encoder.latent_dim
        return self.dynamic.prior_sampling((batch_size, num_latents, latent_dim)).to(self.device)

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):
        del prompt_tokens, prompt_mask
        self._ensure_encoder_decoder_device()
        self._eval_mode()
        self.encoder_decoder.eval()

        if num_steps is None:
            num_steps = self.config.algo.dynamic.N

        z_t = self._sample_latent_prior(num_samples)
        x_0_self_cond = torch.zeros_like(z_t)

        diffusion_cfg = self.config.algo.diffusion
        timesteps = torch.linspace(
            diffusion_cfg.T,
            diffusion_cfg.t_min,
            num_steps + 1,
            device=self.device,
        )

        z_mean = z_t
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(num_samples, device=self.device)
            next_t = timesteps[i + 1] * torch.ones(num_samples, device=self.device)

            output = self.diff_eq_solver.step(
                x_t=z_t,
                t=t,
                next_t=next_t,
                x_0_self_cond=x_0_self_cond,
            )
            z_t = output["x"]
            z_mean = output["x_mean"]
            x_0_self_cond = output["x_0"]

        pred_latent = self.denormalize_latent(z_mean)
        logp_x0 = self._decode_latent(pred_latent)

        self._train_mode()
        return logp_x0.argmax(dim=-1)
