"""Original corruption/self-conditioning recipes across selected battery tasks."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps.configuration import ROOT, compose
from examples.task_battery import build_jobs as battery_jobs
from examples.corruption_self_conditioning_sweep import (
    parser as original_parser,
    variants,
    base_config,
)


def build_jobs(group, *, tasks=None, project="self-cond-sweep"):
    battery = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/battery/small_tasks.yaml")
    )
    battery["name"] = "self_conditioning_battery_20m"
    battery["seeds"] = [1]
    if tasks is not None:
        unknown = set(tasks) - {t["name"] for t in battery["tasks"]}
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        battery["tasks"] = [t for t in battery["tasks"] if t["name"] in tasks]
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
            "trainer.val_check_interval": 4000,
            "trainer.precision": "bf16-mixed",
            "loader.eval_batch_size": 8,
        }
    )
    jobs = []
    args = original_parser().parse_args([])
    for variant in variants(args):
        base = base_config(variant, args)
        for enabled in (False, True):
            name = variant.name + ("-sc-on" if enabled else "-sc-off")
            algorithm = OmegaConf.to_container(
                OmegaConf.load(ROOT / "configs/algorithm_definition/masked.yaml")
            )
            algorithm["name"] = name + "_20m"
            algorithm["components"]["hybrid_view@model.views.content"] = variant.family
            algorithm["settings"].update(
                {
                    "model.backbone.hidden_size": 512,
                    "model.backbone.n_blocks": 6,
                    "model.backbone.n_heads": 8,
                    "model.backbone.cond_dim": 64,
                    "model.views.content": OmegaConf.to_container(
                        base.model.views.content, resolve=False
                    ),
                    "self_conditioning.enabled": enabled,
                    "self_conditioning.source": "hidden",
                    "self_conditioning.probability": 0.9,
                    "self_conditioning.evaluation_probability": 1.0,
                }
            )
            if variant.family == "blurring":
                # Keep terminal leakage below the bound through 1024-token spans.
                algorithm["settings"]["model.views.content.schedule.terminal_mean"] = (
                    0.01
                )
            for job in battery_jobs(algorithm, battery):
                label = f"{name}-20m-{job.name}"
                job.config["wandb"] = {
                    "project": project,
                    "entity": "rz-text-dlp",
                    "group": group,
                    "name": label + "/train",
                    "job_type": "train",
                }
                periodic = {}
                for policy in algorithm["policies"]:
                    sampler = (
                        compose(overrides={"sampler": policy["sampler"]}).sampler
                        if policy.get("sampler")
                        else OmegaConf.create(job.config["sampler"])
                    )
                    for key, value in policy.get("overrides", {}).items():
                        OmegaConf.update(sampler, key.removeprefix("sampler."), value)
                    periodic[policy["name"]] = OmegaConf.to_container(
                        sampler, resolve=False
                    )
                job.config["callback"]["sampler_configs"] = periodic
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
    parser.add_argument("--name", default="three-tasks-20m-bf16-20261012")
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore this Ray experiment and its submission queue",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Use saved training checkpoints from an interrupted experiment",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sudoku_hard", "text8", "tinystories"],
        choices=["sudoku_hard", "nqueens", "text8", "tinystories", "maze"],
    )
    parser.add_argument("--wandb-project", default="self-cond-three-tasks-20261012")
    parser.add_argument("--include-insertion", action="store_true")
    args = parser.parse_args()
    if args.restore and args.resume_from:
        parser.error("Use --restore or --resume-from, not both")
    jobs = (
        []
        if args.restore
        else build_jobs(args.name, tasks=args.tasks, project=args.wandb_project)
    )
    if args.include_insertion and not args.restore:
        from examples.insertion_head_sweep import build_jobs as insertion_jobs

        jobs.extend(
            insertion_jobs(
                args.name + "-insertion",
                tasks=args.tasks or ("text8", "tinystories", "maze"),
                project=args.wandb_project,
            )
        )
    if args.resume_from:
        checkpoints = {}
        for params in args.resume_from.glob("*/params.json"):
            import json

            previous = json.loads(params.read_text())
            checkpoint = params.parent / "run/checkpoints/last.ckpt"
            if previous.get("stage") == "train" and checkpoint.is_file():
                checkpoints[previous["name"]] = checkpoint.resolve()
        for job in jobs:
            if job.name in checkpoints:
                job.config["checkpoint"]["resume"] = str(checkpoints[job.name])
                print(f"Resume {job.name}: {checkpoints[job.name]}", flush=True)
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
    from core.sweeps.submissions import SubmissionQueue

    storage = Path.home() / ".cache/duo/batteries"
    if (storage / args.name).exists() and not args.restore:
        parser.error("Experiment already exists; restore with Ray or use a new name")
    temporary = Path.home() / ".cache/duo/ray"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("WANDB_DATA_DIR", str(Path.home() / ".cache/duo/wandb-data"))
    ray.init(_temp_dir=str(temporary))
    try:
        trainable = tune.with_resources(run_job, {"cpu": 4, "gpu": 1})
        if args.restore:
            tuner = tune.Tuner.restore(str(storage / args.name), trainable=trainable)
        else:
            queue = SubmissionQueue.create(
                storage / "submissions" / f"{args.name}.sqlite"
            )
            queue.submit(jobs)
            print(f"Live submission queue: {queue.path}", flush=True)
            tuner = tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    search_alg=TrainingInferenceSearch(submission_path=queue.path),
                    num_samples=-1,
                    max_concurrent_trials=4,
                ),
                run_config=tune.RunConfig(
                    storage_path=str(storage), name=args.name, log_to_file=True
                ),
            )
        results = tuner.fit()
        if results.num_errors:
            raise RuntimeError(f"{results.num_errors} trials failed; inspect Ray logs")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
