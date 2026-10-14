"""Seven approximately 20M-parameter algorithms across the five-task battery."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps.configuration import ROOT
from examples.task_battery import build_jobs as battery_jobs


def build_jobs(group):
    battery = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/battery/small_tasks.yaml")
    )
    battery["name"] = "full_battery_20m"
    battery["seeds"] = [1]
    for task in battery["tasks"]:
        if task["name"] == "text8":
            task["settings"]["data.cache_dir"] = (
                "/home/patrick/.cache/duo/worktrees/22a3/data_cache/text8"
            )
        if task["name"] == "tinystories":
            task["settings"]["task.calculate_bleu"] = True
    battery["settings"].update(
        {
            "trainer.accumulate_grad_batches": 4,
            "loader.eval_batch_size": 8,
        }
    )
    jobs = []
    for name in (
        "masked",
        "masked_sc",
        "masked_sc_insert",
        "uniform",
        "uniform_sc",
        "uniform_sc_insert",
        "ar",
    ):
        algorithm = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/algorithm_definition/masked.yaml")
        )
        algorithm["name"] = name + "_20m"
        algorithm["settings"].update(
            {
                "model.backbone.hidden_size": 512,
                "model.backbone.n_blocks": 6,
                "model.backbone.n_heads": 8,
                "self_conditioning.enabled": "_sc" in name,
                "self_conditioning.source": "posterior_mean",
                "self_conditioning.probability": 0.9,
                "self_conditioning.evaluation_probability": 1.0,
            }
        )
        if name.startswith("uniform"):
            algorithm["components"]["hybrid_view@model.views.content"] = "uniform"
        if name.endswith("_insert"):
            algorithm["components"].update(
                {"variable_length": "insertion", "sampler": "insertion"}
            )
            algorithm["settings"].update(
                {
                    "sampler.max_length": "${task.length}",
                    "sampler.overflow": "truncate",
                }
            )
            algorithm["policies"] = [
                {
                    "name": "insertion_128",
                    "overrides": {"sampler.final_denoise": "sample"},
                },
                {
                    "name": "insertion_128_greedy",
                    "overrides": {"sampler.final_denoise": "greedy"},
                },
            ]
        if name == "ar":
            algorithm["components"] = {
                "algorithm": "autoregressive",
                "model": "causal_dit",
                "parameterization": "autoregressive",
                "objective": "cross_entropy",
                "sampler": "autoregressive",
            }
            algorithm["settings"] = {
                "model.hidden_size": 512,
                "model.n_blocks": 6,
                "model.n_heads": 8,
                "model.dropout": 0.1,
                "self_conditioning.enabled": False,
            }
            algorithm["policies"] = [{"name": "ar_sample", "overrides": {}}]
        for job in battery_jobs(algorithm, battery):
            label = f"{name}-20m-{job.name}"
            job.config["wandb"] = {
                "project": "task-battery",
                "entity": "rz-text-dlp",
                "group": group,
                "name": label + "/train",
                "job_type": "train",
            }
            inference = tuple(
                replace(
                    choice,
                    overrides={
                        **choice.overrides,
                        "wandb.name": label
                        + "/eval/"
                        + choice.overrides["evaluation.policy"],
                        "wandb.job_type": "evaluate",
                    },
                )
                for choice in job.inference
            )
            jobs.append(replace(job, name=label, inference=inference))
    return sorted(
        jobs, key=lambda job: (job.config["benchmark"]["task_id"], job.config["seed"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--name", default="full-battery-20m-20261012")
    args = parser.parse_args()
    jobs = build_jobs(args.name)
    from core.training.module import TrainingModule

    for job in jobs:
        preview = OmegaConf.create(job.config)
        preview.training.ema = 0
        module = TrainingModule(preview)
        count = sum(p.numel() for p in module.algorithm.parameters())
        job.config["benchmark"]["parameter_count"] = count
        print(f"{job.name}: {count:,} parameters; 10,000 steps", flush=True)
        del module
    if not args.run:
        return
    import ray
    from ray import tune
    from core.sweeps.queue import TrainingInferenceSearch, run_job

    storage = Path.home() / ".cache/duo/batteries"
    if (storage / args.name).exists():
        parser.error("Experiment already exists; restore with Ray or use a new name")
    temporary = Path.home() / ".cache/duo/ray"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("WANDB_DATA_DIR", str(Path.home() / ".cache/duo/wandb-data"))
    ray.init(_temp_dir=str(temporary))
    try:
        results = tune.Tuner(
            tune.with_resources(run_job, {"cpu": 4, "gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=TrainingInferenceSearch(jobs),
                num_samples=-1,
                max_concurrent_trials=4,
            ),
            run_config=tune.RunConfig(
                storage_path=str(storage), name=args.name, log_to_file=True
            ),
        ).fit()
        if results.num_errors:
            raise RuntimeError(f"{results.num_errors} trials failed; inspect Ray logs")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
