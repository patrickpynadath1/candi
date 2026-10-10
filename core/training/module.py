"""Lightning lifecycle, deliberately separate from MDLM mathematics."""

from contextlib import contextmanager
import random
import numpy as np
import lightning as L
from omegaconf import OmegaConf
import torch

from hydra.utils import instantiate
from .ema import ExponentialMovingAverage


class TrainingModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        from core.modifiers.hybrid.checkpoint import upgrade_config

        raw = (
            OmegaConf.to_container(config, resolve=False)
            if OmegaConf.is_config(config)
            else config
        )
        self.config = OmegaConf.create(upgrade_config(raw))
        self.save_hyperparameters(
            {"config": OmegaConf.to_container(self.config, resolve=True)}
        )
        self.algorithm = instantiate(self.config.algorithm)
        structural = self.config.get("variable_length")
        if structural is not None:
            self.algorithm = instantiate(structural, algorithm=self.algorithm)
        block = self.config.get("block_diffusion", {})
        if block.get("enabled", False):
            from core.modifiers.block_diffusion import BlockDiffusion

            self.algorithm = BlockDiffusion(
                self.algorithm,
                block_size=block.block_size,
                clean_loss=block.get("clean_loss", "none"),
                clean_loss_weight=block.get("clean_loss_weight", 1.0),
                cache_clean_prefix=block.get("cache_clean_prefix", True),
                max_length=block.get("max_length"),
            )
        feedback = self.config.get("token_feedback", {})
        if feedback.get("enabled", False):
            from core.modifiers.token_feedback import TokenFeedback

            self.algorithm = TokenFeedback(
                self.algorithm,
                depth=feedback.depth,
                loss_weight=feedback.get("loss_weight", 1.0),
                gate=feedback.get("gate", "scalar"),
                detach=feedback.get("detach", False),
            )
        if self.config.get("self_conditioning", {}).get("enabled", False):
            policy = getattr(self.algorithm, "prediction", None)
            if policy is None or not policy.enabled:
                raise ValueError(
                    "self-conditioning requires an enabled prediction coordinator on this algorithm"
                )
        self.ema = (
            ExponentialMovingAverage(
                self.algorithm.parameters(), self.config.training.ema
            )
            if self.config.training.ema > 0
            else None
        )
        self._pending_rng = None
        self._train_metric_totals = {}

    def on_fit_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    def on_train_start(self):
        if self._pending_rng:
            state = self._pending_rng
            torch.set_rng_state(state["torch"].cpu())
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            if self.device.type == "cuda" and state.get("cuda") is not None:
                torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)
            self._pending_rng = None

    def training_step(self, batch, batch_idx):
        result = self.algorithm.loss(
            batch["input_ids"],
            batch["loss_mask"],
            corruption_mask=batch.get("corruption_mask"),
            **(
                {"valid_mask": batch.get("attention_mask")}
                if getattr(self.algorithm, "supports_variable_length", False)
                else {}
            ),
            step=self.global_step,
            training=True,
            global_batch_size=self.config.loader.global_batch_size,
            world_size=self.trainer.world_size,
            rank=self.global_rank,
            accumulation_steps=self.trainer.accumulate_grad_batches,
            accumulation_index=batch_idx % self.trainer.accumulate_grad_batches,
        )
        # Accumulate detached, token-weighted metrics across microbatches.
        # Logging only the last microbatch aliases the stratified time ranges.
        count = result["num_tokens"].detach().float()
        for key in (
            "loss",
            "main_loss",
            "token_feedback_loss",
            "content_loss",
            "insertion_loss",
            "insertion_content_loss",
        ):
            if key in result:
                value = result[key].detach().float() * count
                previous_sum, previous_count = self._train_metric_totals.get(
                    key, (0, 0)
                )
                self._train_metric_totals[key] = (
                    previous_sum + value,
                    previous_count + count,
                )
        return result["loss"]

    def on_before_optimizer_step(self, optimizer):
        for name, (total, count) in self._train_metric_totals.items():
            self.log(
                f"train/{name}",
                total / count.clamp_min(1),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        self._train_metric_totals.clear()

    @contextmanager
    def evaluation_weights(self, use_ema=True):
        was_training = self.algorithm.training
        if use_ema and self.ema:
            self.ema.move_shadow_params_to_device(self.device)
            self.ema.store(self.algorithm.parameters())
            self.ema.copy_to(self.algorithm.parameters())
        self.algorithm.eval()
        try:
            yield
        finally:
            if use_ema and self.ema:
                self.ema.restore(self.algorithm.parameters())
            self.algorithm.train(was_training)

    def on_validation_epoch_start(self):
        self._validation_context = self.evaluation_weights()
        self._validation_context.__enter__()
        self._valid_sum = torch.zeros((), device=self.device)
        self._valid_count = torch.zeros((), device=self.device)

    def validation_step(self, batch, batch_idx):
        result = self.algorithm.loss(
            batch["input_ids"],
            batch["loss_mask"],
            corruption_mask=batch.get("corruption_mask"),
            **(
                {"valid_mask": batch.get("attention_mask")}
                if getattr(self.algorithm, "supports_variable_length", False)
                else {}
            ),
        )
        self._valid_sum += result["loss_sum"]
        self._valid_count += result["num_tokens"]

    def on_validation_epoch_end(self):
        totals = (
            self.all_gather(torch.stack([self._valid_sum, self._valid_count]))
            .reshape(-1, 2)
            .sum(0)
        )
        self.log("val/nll", totals[0] / totals[1].clamp_min(1), sync_dist=False)
        self._validation_context.__exit__(None, None, None)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self.algorithm.parameters())
        # Match the reference lifecycle: EMA sees the optimizer's update,
        # then representations enforce their parameter constraints.
        for component in self.algorithm.modules():
            hook = getattr(component, "post_optimizer_step", None)
            if callable(hook):
                hook()

    def optimizer_parameters(self, config):
        """Optional parameter-name groups, validated to avoid overlap/typos."""
        import fnmatch

        groups = config.get("groups", [])
        if not groups:
            return self.algorithm.parameters()
        named = dict(self.algorithm.named_parameters())
        used, result = set(), []
        for group in groups:
            names = [
                name
                for name in named
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in group.patterns)
            ]
            if not names or used.intersection(names):
                raise ValueError(
                    "optimizer groups must match parameters without overlap"
                )
            for pattern in group.patterns:
                if not any(fnmatch.fnmatchcase(name, pattern) for name in names):
                    raise ValueError(
                        f"optimizer pattern {pattern!r} matches no parameters"
                    )
            used.update(names)
            options = {key: value for key, value in group.items() if key != "patterns"}
            result.append({"params": [named[name] for name in names], **options})
        remaining = [parameter for name, parameter in named.items() if name not in used]
        return [{"params": remaining}, *result] if remaining else result

    def configure_optimizers(self):
        c = self.config.optimizer
        optimizer = torch.optim.AdamW(
            self.optimizer_parameters(c),
            lr=c.lr,
            betas=(c.beta1, c.beta2),
            eps=c.epsilon,
            weight_decay=c.weight_decay,
        )
        if self.config.get("lr_scheduler") is not None:
            scheduler = instantiate(self.config.lr_scheduler, optimizer=optimizer)
        else:
            # Saved runs predating the scheduler config group retain their recipe.
            from .schedules import constant_warmup

            scheduler = constant_warmup(optimizer, self.config.training.warmup_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["core_format_version"] = 1
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        checkpoint["rng"] = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
        }

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            if "ema" not in checkpoint:
                raise ValueError("EMA-enabled resume requires saved EMA state")
            self.ema.load_state_dict(checkpoint["ema"])
        self._pending_rng = checkpoint.get("rng")
