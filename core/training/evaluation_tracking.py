"""Publish completed standalone evaluations to the configured W&B project."""

import hashlib
import json

from omegaconf import OmegaConf

from core.evaluation.reporting import scalar_metrics


def log_evaluation(config, result, output):
    if config.get("wandb") is None:
        return
    import wandb

    settings = dict(config.wandb)
    parent_name = settings.pop("name", "model")
    sampler = OmegaConf.to_container(config.sampler, resolve=True)
    signature = hashlib.sha256(
        json.dumps(sampler, sort_keys=True).encode()
    ).hexdigest()[:10]
    settings["job_type"] = "evaluate"
    settings.pop("id", None)  # Never reuse the parent training run's identity.
    with wandb.init(
        **settings,
        name=f"{parent_name}-eval-{signature}",
        dir=str(output),
        config=OmegaConf.to_container(config, resolve=True),
    ) as run:
        metrics = dict(scalar_metrics(result))
        run.log(metrics)
        run.summary.update(metrics)
        run.summary["parent_training_name"] = parent_name
        run.summary["results_path"] = str(output / "results.json")
