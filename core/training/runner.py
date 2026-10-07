"""Shared training, evaluation, and sampling lifecycle."""

import json
from pathlib import Path
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf

from core.training.checkpoints import load_weights
from core.training.module import TrainingModule
from core.training.strategies import build_strategy


def run(config):
    L.seed_everything(config.seed, workers=True)
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Persist a portable resolved config alongside every run's outputs.
    OmegaConf.save(
        OmegaConf.create(OmegaConf.to_container(config, resolve=True)),
        output / "config.yaml",
    )
    task = instantiate(config.task)
    data = instantiate(config.datamodule, task=task)
    module = TrainingModule(config)
    if config.mode == "train":
        callbacks = [
            ModelCheckpoint(
                dirpath=output / "checkpoints",
                every_n_train_steps=config.checkpoint.every_n_steps,
                every_n_epochs=config.checkpoint.get("every_n_epochs", None),
                save_last=True,
                save_top_k=config.checkpoint.get("save_top_k", -1),
                monitor="step" if config.checkpoint.get("save_top_k", -1) > 0 else None,
                mode="max",
                filename="step_{step:08d}",
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                dirpath=output / "checkpoints",
                monitor="val/nll",
                mode="min",
                save_top_k=1,
                filename="best",
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        if config.get("callback") is not None:
            callbacks.append(instantiate(config.callback, task=task))
        loggers = [CSVLogger(str(output), name="metrics")]
        tracking = None
        if config.wandb is not None:
            tracking = WandbLogger(
                save_dir=str(output),
                config=OmegaConf.to_container(config, resolve=True),
                **dict(config.wandb),
            )
            loggers.append(tracking)
        trainer_options = OmegaConf.to_container(config.trainer, resolve=True)
        trainer_options["strategy"] = build_strategy(config, module.algorithm)
        trainer = L.Trainer(
            default_root_dir=str(output),
            logger=loggers,
            callbacks=callbacks,
            **trainer_options,
        )
        tracking_run = None
        exit_code = 1
        try:
            if tracking is not None and trainer.is_global_zero:
                tracking_run = tracking.experiment
            trainer.fit(module, datamodule=data, ckpt_path=config.checkpoint.resume)
            final_checkpoint = output / "checkpoints" / "last.ckpt"
            trainer.save_checkpoint(final_checkpoint)
            if tracking_run is not None:
                tracking_run.summary["final_checkpoint"] = str(final_checkpoint)
                tracking_run.summary["completed_training_steps"] = trainer.global_step
            exit_code = 0
        finally:
            if tracking_run is not None:
                tracking_run.finish(exit_code=exit_code)
    elif config.mode in {"evaluate", "sample"}:
        if not config.evaluation.checkpoint:
            raise ValueError(
                "evaluation.checkpoint is required for evaluation or sampling"
            )
        if not config.evaluation.use_ema:
            module.ema = None
        step = load_weights(
            module, config.evaluation.checkpoint, legacy=config.evaluation.legacy
        )
        module.to(config.evaluation.device)
        task.prepare_evaluation(task.primary_metric.dataset)
        evaluate = instantiate(config.evaluator)
        result = evaluate(
            module,
            task,
            config.sampler,
            batch_size=config.loader.eval_batch_size,
            max_examples=config.evaluation.max_examples,
            use_ema=config.evaluation.use_ema,
            precision=config.trainer.precision,
        )
        result["primary_metric"] = {
            "name": task.primary_metric.name,
            "mode": task.primary_metric.mode,
            "dataset": task.primary_metric.dataset,
        }
        result["checkpoint_step"] = step
        result["seed"] = config.seed
        result["parameterization"] = OmegaConf.to_container(
            config.parameterization, resolve=True
        )
        result["sampler"] = OmegaConf.to_container(config.sampler, resolve=True)
        (output / "results.json").write_text(json.dumps(result, indent=2))
        from core.training.evaluation_tracking import log_evaluation

        log_evaluation(config, result, output)
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    else:
        raise ValueError("mode must be train, evaluate or sample")
