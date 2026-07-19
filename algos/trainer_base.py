import itertools
import json
from dataclasses import dataclass
from pathlib import Path
import hydra.utils
import lightning as L
import torch
import transformers

import dataloader
import metrics
from models import model_registry
from models.ema import ExponentialMovingAverage
from .noise_schedules import LogLinear, get_noise

from samplers import SamplingRequest, create_sampler, run_sampler

@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    prior_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor


def sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
    default_sampler = None
    supported_samplers = frozenset()
    def __init__(
        self, config, tokenizer: transformers.PreTrainedTokenizer, vocab_size=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if hasattr(self.config.algo, "ignore_bos"):
            self.ignore_bos = config.algo.ignore_bos
        else:
            self.ignore_bos = False
        if hasattr(self.config.algo, "loss_type"):
            self.loss_type = config.algo.loss_type
        self.tokenizer = tokenizer
        if vocab_size is None:
            if hasattr(self.config.data, 'vocab_file'): 
                self.vocab_size = len(self.tokenizer.added_tokens_decoder)
                self.token_offset = True
            else: 
                self.token_offset = False
                self.vocab_size = len(self.tokenizer)
        else:
            self.vocab_size = vocab_size

        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.parameterization = self.config.algo.parameterization

        if self.config.algo.backbone == "hf_dit":
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True
            )
        else:
            backbone_cls = model_registry.get(self.config.algo.backbone)
            self.backbone = backbone_cls(self.config, vocab_size=self.vocab_size)

        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.softplus = torch.nn.Softplus()
        self.p_nucleus = self.config.sampling.p_nucleus
        # All time-conditioned algorithms use the configured path schedule.
        self.noise = get_noise(config)
        self._last_logged_schedule_refit = {}

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
            eval_model_source=self.config.eval.get("eval_model_source", "hf"),
            local_eval_checkpoint_path=self.config.eval.get(
                "local_eval_checkpoint_path", None
            ),
            local_eval_device=self.config.eval.get("local_eval_device", "cuda"),
            local_eval_context_size=self.config.eval.get(
                "local_eval_context_size", None
            ),
        )

        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema
            )
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

    def _validate_configuration(self):
        assert self.config.algo.backbone in {"dit", "hf_dit"}
        if self.config.algo.parameterization == "ar":
            assert not self.config.algo.time_conditioning
            assert self.config.prior.type == "none"

        if self.parameterization in {"score", "mean"}:
            assert self.time_conditioning
        if self.T > 0:
            assert self.parameterization != "score"

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return self

    def q_xt(self, x, alpha_t):
        raise NotImplementedError

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(), self.noise.parameters())

    def _eval_mode(self):
        if self.ema:
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        if self.ema:
            self.ema.restore(self._get_parameters())
        self.backbone.train()
        self.noise.train()

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"][
            "current"
        ]["completed"]
        self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
            "epoch_loop.batch_progress"
        ]["current"]["completed"]

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            cleaned_state_dict = {}
            
            for key, value in state_dict.items():
                if "teacher" not in key: 
                    cleaned_key = key
                    # Remove _orig and other modifiers
                    cleaned_key = cleaned_key.replace('_orig_mod.', '') 
                    cleaned_state_dict[cleaned_key] = value
            
            checkpoint['state_dict'] = cleaned_state_dict


    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed']
        # is 1 iteration behind, so we're using the optimizer's progress.
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["total"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["current"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        # _batches_that_stepped tracks the number of global steps,
        # not the number of local steps, so we don't multiply with
        # self.trainer.accumulate_grad_batches here.
        checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
            "_batches_that_stepped"
        ] = checkpoint["loops"]["fit_loop"][
            "epoch_loop.automatic_optimization.optim_progress"
        ][
            "optimizer"
        ][
            "step"
        ][
            "total"
        ][
            "completed"
        ]
        if "sampler" not in checkpoint.keys():
            checkpoint["sampler"] = {}
        if hasattr(self.trainer.train_dataloader.sampler, "state_dict"):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint["sampler"]["random_state"] = sampler_state_dict.get(
                "random_state", None
            )
        else:
            checkpoint["sampler"]["random_state"] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
        )
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, "shuffle"):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (
                distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                dl_sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": (
                            self.fast_forward_batches * self.config.loader.batch_size
                        ),
                    }
                )
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
            persistent_workers=self.config.loader.num_workers > 0,
                )
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    def _record_time_loss(
        self, t, token_loss, valid_tokens, train_mode,
        schedule=None, schedule_name="schedule",
    ):
        schedule = self.noise if schedule is None else schedule
        if not train_mode or not hasattr(schedule, "record_time_loss_pair"):
            return
        if valid_tokens is None:
            per_example = token_loss.detach().mean(-1)
        else:
            weights = valid_tokens.to(token_loss)
            per_example = (token_loss.detach() * weights).sum(-1) / weights.sum(
                -1
            ).clamp_min(1)
        schedule.record_time_loss_pair(t.detach(), per_example, int(self.global_step))
        self._log_adaptive_schedule(schedule, schedule_name)

    def _log_adaptive_schedule(self, schedule=None, schedule_name="schedule"):
        schedule = self.noise if schedule is None else schedule
        snapshot_fn = getattr(schedule, "snapshot", None)
        if snapshot_fn is None:
            return
        refit = int(getattr(schedule, "refit_count").item())
        last_refit = self._last_logged_schedule_refit.get(schedule_name, 0)
        if refit <= last_refit or self.global_rank != 0:
            return
        self._last_logged_schedule_refit[schedule_name] = refit
        snapshot = snapshot_fn()
        output_dir = Path(
            getattr(
                self.config.noise,
                "adaptive_snapshot_dir",
                Path(self.config.checkpointing.save_dir) / "schedule_snapshots",
            )
        )
        if schedule_name != "schedule":
            output_dir = output_dir / schedule_name
        output_dir.mkdir(parents=True, exist_ok=True)
        step = int(self.global_step)
        (output_dir / f"step_{step:06d}.json").write_text(
            json.dumps(snapshot, indent=2)
        )
        try:
            import wandb

            experiment = self.logger.experiment
            plot = wandb.plot.line_series(
                xs=snapshot["grid"],
                ys=[
                    snapshot["base_alpha"],
                    snapshot["alpha"],
                    snapshot["sampling_density"],
                ],
                keys=["base_alpha", "adaptive_alpha", "sampling_density"],
                title=f"Adaptive schedule at step {step}",
                xname="t",
            )
            dynamics_plot = wandb.plot.line_series(
                xs=snapshot["grid"],
                ys=[snapshot["loss_profile"], snapshot["dalpha_dt"]],
                keys=["smoothed_loss", "dalpha_dt"],
                title=f"Adaptive schedule dynamics at step {step}",
                xname="t",
            )
            experiment.log(
                {
                    f"{schedule_name}/profile": plot,
                    f"{schedule_name}/dynamics": dynamics_plot,
                    f"{schedule_name}/refit_count": refit,
                    f"{schedule_name}/alpha_min": min(snapshot["alpha"]),
                    f"{schedule_name}/alpha_max": max(snapshot["alpha"]),
                },
                step=step,
            )
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _process_sigma(self, sigma):
        raise NotImplementedError

    def _process_model_output(self, model_output, xt, sigma):
        raise NotImplementedError

    def forward(self, xt, sigma):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(xt, sigma)
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    def on_train_epoch_start(self):
        self.metrics.reset()
        assert self.metrics.train_nlls.nll.mean_value == 0
        assert self.metrics.train_nlls.nll.weight == 0

    def training_step(self, batch, batch_idx):

        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            current_accumulation_step,
            train_mode=True,
        )
        self.metrics.update_train(losses.nlls, losses.prior_loss, losses.num_tokens)
        self.log(
            name="trainer/loss",
            value=losses.loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return losses.loss

    def on_train_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name=k, value=v.compute(), on_step=False, on_epoch=True, sync_dist=True
            )

    def on_validation_epoch_start(self):
        self.metrics.reset()
        self._eval_mode()
        assert self.metrics.valid_nlls.nll.mean_value == 0
        assert self.metrics.valid_nlls.nll.weight == 0

    def validation_step(self, batch, batch_idx):

        del batch_idx
        losses = self._loss(batch["input_ids"], batch["attention_mask"])
        self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
        return losses.loss

    def _validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name=k, value=v.compute(), on_step=False, on_epoch=True, sync_dist=True
            )
        if (
            self.config.eval.compute_perplexity_on_sanity
            or not self.trainer.sanity_checking
        ) and self.config.eval.generate_samples:
            samples, text_samples = None, None
            for _ in range(self.config.sampling.num_sample_batches):
                samples = self.generate_samples(
                    num_samples=self.config.loader.eval_batch_size
                )

                self.metrics.record_entropy(samples)
                # Decode the samples to be re-tokenized by eval model
                text_samples = self.tokenizer.batch_decode(samples)
                if self.config.eval.compute_generative_perplexity:
                    self.metrics.record_generative_perplexity(
                        text_samples, self.num_tokens, self.device
                    )
            if text_samples is not None:
                if self.trainer.global_rank == 0 and hasattr(
                    self.trainer.logger, "log_table"
                ):
                    # Log the last generated samples
                    text_samples = text_samples[: self.config.sampling.num_sample_log]
                    self.trainer.logger.log_table(
                        key=f"samples@global_step{self.global_step}",
                        columns=["Generated Samples"],
                        data=[[s] for s in text_samples],
                    )
                if self.config.eval.compute_generative_perplexity:
                    self.log(
                        "val/gen_ppl",
                        self.metrics.gen_ppl.compute(),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )
                    self.log(
                        "val/gen_bpc",
                        self.metrics.gen_bpc.compute(),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )
                    self.log(
                        "val/sample_entropy",
                        self.metrics.sample_entropy.compute(),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )
    def on_validation_epoch_end(self):
        try:
            self._validation_epoch_end()
        finally:
            self._train_mode()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def generate_samples(
        self, num_samples, num_steps=None, eps=1e-5,
        prompt_tokens=None, prompt_mask=None, **options
    ):
        configured = getattr(self.config.sampling, "predictor", None)
        sampler_name = (
            configured if configured in self.supported_samplers else self.default_sampler
        )
        legacy_method = options.pop(
            "sampling_method", getattr(self, "sampling_method", None)
        )
        if legacy_method is not None and self.sampling_family == "semicat":
            sampler_name = f"semicat_{legacy_method}"
            if sampler_name not in self.supported_samplers:
                raise ValueError(f"Unknown sampling method: {legacy_method}")
        sampler = create_sampler(self.config, default=sampler_name)
        result = run_sampler(
            sampler,
            self,
            SamplingRequest(
                num_samples=num_samples,
                num_steps=num_steps,
                eps=eps,
                prompt_tokens=prompt_tokens,
                prompt_mask=prompt_mask,
                options=options,
            ),
        )
        return result.samples

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self._eval_mode()
        try:
            return self.generate_samples(
                num_samples=self.config.loader.eval_batch_size,
                num_steps=num_steps,
                eps=eps,
            )
        finally:
            self._train_mode()

    def _process_model_input(self, x0, valid_tokens):
        raise NotImplementedError

    def nll(
        self,
        input_tokens,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        raise NotImplementedError

    def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        (input_tokens, output_tokens, valid_tokens) = self._process_model_input(
            x0, valid_tokens
        )
        loss = self.nll(
            input_tokens, output_tokens, current_accumulation_step, train_mode,
            valid_tokens=valid_tokens,
        )
        assert loss.ndim == 2
        if self.ignore_bos:
            valid_tokens = valid_tokens.clone()
            valid_tokens[:, 0] = 0

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens

        return Loss(loss=token_nll, nlls=nlls, prior_loss=0.0, num_tokens=num_tokens)
