"""Shared training, evaluation, and sampling lifecycle."""

import json
from pathlib import Path
from hydra.utils import instantiate
import lightning as L
import logging
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf

from core.training.checkpoints import load_weights
from core.training.module import TrainingModule
from core.training.strategies import build_strategy


def run(config):
    # Default also applies to queued configs created before compilation existed.
    if "compile" not in config:
        config.compile = True
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
    # Explicit backbone compilation already covers its transformer blocks.
    backbone_compiled = OmegaConf.select(
        config, "training.performance.compile.enabled", default=False
    )
    if config.compile and not backbone_compiled:
        from core.models.layers import DDiTBlock

        blocks = [block for block in module.modules() if isinstance(block, DDiTBlock)]
        for block in blocks:
            # In-place compilation preserves parameter names and checkpoint keys.
            block.compile(dynamic=False)
        logging.getLogger(__name__).info(
            "torch.compile enabled for %d transformer blocks", len(blocks)
        )
    if config.mode == "train":
        callbacks = [
            ModelCheckpoint(
                dirpath=output / "checkpoints",
                every_n_train_steps=config.checkpoint.every_n_steps,
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
        evaluation_name = (
            config.evaluator.get("evaluation_name") or task.primary_metric.dataset
        )
        task.prepare_evaluation(evaluation_name)
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
            "dataset": evaluation_name,
        }
        result["checkpoint_step"] = step
        if config.get("benchmark") is not None:
            result["benchmark"] = OmegaConf.to_container(config.benchmark, resolve=True)
        result["seed"] = config.seed
        result["parameterization"] = OmegaConf.to_container(
            config.parameterization, resolve=True
        )
        result["sampler"] = OmegaConf.to_container(config.sampler, resolve=True)
        (output / "results.json").write_text(json.dumps(result, indent=2))
        if config.wandb is not None:
            import wandb
            from core.evaluation.reporting import scalar_metrics

            policy = config.evaluation.get("policy")
            namespace = f"{task.name}/{policy}" if policy else task.name
            options = dict(config.wandb)
            options["job_type"] = "evaluate"
            with wandb.init(
                dir=str(output),
                config=OmegaConf.to_container(config, resolve=True),
                **options,
            ) as tracking_run:
                tracking_run.log(
                    {
                        **{
                            f"{namespace}/{key}": value
                            for key, value in scalar_metrics(result)
                        },
                        "trainer/global_step": step,
                    }
                )
                artifact = wandb.Artifact(
                    f"{task.name}-{policy or 'final'}-evaluation", type="evaluation"
                )
                artifact.add_file(str(output / "results.json"))
                tracking_run.log_artifact(artifact)
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    else:
        raise ValueError("mode must be train, evaluate or sample")
