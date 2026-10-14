"""Submit the discrete insertion gap-head × self-conditioning ablation to Ray."""

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps.configuration import ROOT
from core.sweeps.submissions import SubmissionQueue
from examples.task_battery import build_jobs as battery_jobs


GROUP = "three-tasks-20m-bf16-20261012-insertion"
QUEUE = (
    Path.home()
    / ".cache/duo/batteries/submissions/three-tasks-20m-bf16-20261012.sqlite"
)


def build_jobs(
    group=GROUP, *, tasks=("text8", "tinystories", "maze"), project="self-cond-sweep"
):
    battery = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/battery/small_tasks.yaml")
    )
    battery["name"] = "insertion_gap_head_20m"
    battery["seeds"] = [1]
    unknown = set(tasks) - {t["name"] for t in battery["tasks"]}
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    battery["tasks"] = [t for t in battery["tasks"] if t["name"] in tasks]
    battery["settings"].update(
        {
            "trainer.precision": "bf16-mixed",
            "trainer.accumulate_grad_batches": 4,
            "trainer.val_check_interval": 4000,
            "loader.eval_batch_size": 8,
            "callback.subset_size": 64,
        }
    )
    for task in battery["tasks"]:
        if task["name"] == "text8":
            task["settings"]["data.cache_dir"] = (
                "/home/patrick/.cache/duo/worktrees/22a3/data_cache/text8"
            )
        if task["name"] == "tinystories":
            task["settings"]["task.calculate_bleu"] = True
    jobs = []
    for corruption, head, sc in product(
        ("masked", "uniform"), (False, True), (False, True)
    ):
        name = f"{corruption}-insert-head-{'on' if head else 'off'}-sc-{'on' if sc else 'off'}-20m"
        algorithm = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/algorithm_definition/masked.yaml")
        )
        algorithm["name"] = name
        algorithm["components"].update(
            {
                "hybrid_view@model.views.content": corruption,
                "variable_length": "insertion",
                "sampler": "insertion",
            }
        )
        algorithm["settings"].update(
            {
                "model.backbone.hidden_size": 512,
                "model.backbone.n_blocks": 6,
                "model.backbone.n_heads": 8,
                "model.backbone.cond_dim": 64,
                "self_conditioning.enabled": sc,
                "self_conditioning.source": "hidden",
                "self_conditioning.probability": 0.9,
                "self_conditioning.evaluation_probability": 1.0,
                "variable_length.predict_gap_tokens": head,
                "variable_length.objective.content_weight": 1.0 if head else 0.0,
                "sampler.steps": 127,
                "sampler.max_length": "${task.length}",
                "sampler.overflow": "truncate",
            }
        )
        algorithm["policies"] = [
            {
                "name": "insertion_128_sample",
                "overrides": {"sampler.final_denoise": "sample"},
            },
            {
                "name": "insertion_128_final_greedy",
                "overrides": {"sampler.final_denoise": "greedy"},
            },
        ]
        for job in battery_jobs(algorithm, battery):
            label = f"{name}-{job.name}"
            job.config["wandb"] = {
                "project": project,
                "entity": "rz-text-dlp",
                "group": group,
                "name": label + "/train",
                "job_type": "train",
            }
            periodic = {}
            for policy in algorithm["policies"]:
                sampler = OmegaConf.create(job.config["sampler"])
                for key, value in policy["overrides"].items():
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
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--group", default=GROUP)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sudoku_hard", "text8", "tinystories"],
        choices=["sudoku_hard", "nqueens", "text8", "tinystories", "maze"],
    )
    parser.add_argument("--wandb-project", default="self-cond-three-tasks-20261012")
    args = parser.parse_args()
    from core.training.module import TrainingModule

    jobs = build_jobs(args.group, tasks=args.tasks, project=args.wandb_project)
    for job in jobs:
        config = OmegaConf.create(job.config)
        config.training.ema = 0
        module = TrainingModule(config)
        count = sum(p.numel() for p in module.algorithm.parameters())
        job.config["benchmark"]["parameter_count"] = count
        print(f"{job.name}: {count:,} parameters", flush=True)
        del module
    print(
        f"{len(jobs)} training jobs; {sum(len(j.inference) for j in jobs)} evaluations",
        flush=True,
    )
    if args.submit:
        print(
            f"Accepted {SubmissionQueue(args.queue).submit(jobs)} new jobs", flush=True
        )


if __name__ == "__main__":
    main()
