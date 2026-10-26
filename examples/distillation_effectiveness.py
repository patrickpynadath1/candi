"""Submit matched few-step comparisons to an existing live Ray queue.

Zero-update jobs snapshot the unchanged pretrained weights for baseline evaluation.
Controls continue ordinary base training; procedures start from the same weights.
All comparisons use the same validation examples and fixed inference settings.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps import InferenceJob
from core.sweeps.submissions import SubmissionQueue
from examples.distillation_smoke import build_jobs, discover


def build_comparison(bases, updates=3000, batch_size=16, examples=256):
    if updates < 1 or batch_size < 1 or examples < 1:
        raise ValueError("updates, batch size, and examples must be positive")
    jobs = build_jobs(
        bases,
        updates=updates,
        evaluation_examples=examples,
        recipes=("alpha", "sdtt", "dmmd", "idlm", "dmd"),
    )
    comparisons = []
    for job in jobs:
        cfg = job.config
        cfg.loader.global_batch_size = batch_size
        if "num_eval_samples" in cfg.task:
            cfg.task.num_eval_samples = examples
        cfg.trainer.val_check_interval = min(500, updates)
        cfg.callback.every_n_steps = min(500, updates)
        cfg.callback.subset_size = 32
        cfg.training.warmup_steps = min(100, updates)
        if job.name.startswith("sdtt-"):
            # Six stages: teacher stride 1/128 -> 1/4, two steps per target.
            cfg.procedure.promotion_interval = max(1, updates // 6)
        variants = [(job.name, cfg, (4, 16))]
        # One unchanged baseline and one ordinary-training control per base/task.
        if job.name.startswith(("alpha-", "dmd-")):
            family = "masked" if job.name.startswith("alpha-") else "gaussian"
            suffix = job.name.split("-", 1)[1]
            for label in ("base", "continued"):
                control = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                control.procedure = None
                control.trainer.max_steps = 0 if label == "base" else updates
                control.trainer.gradient_clip_val = 1.0
                control.callback = None if label == "base" else control.callback
                control.checkpoint.every_n_steps = max(1, control.trainer.max_steps)
                variants.append(
                    (
                        f"{family}-{label}-{suffix}",
                        control,
                        (4, 16, 128) if label == "base" else (4, 16),
                    )
                )
        for stem, config, budgets in variants:
            name = f"fewstep-{updates}-{stem}"
            config.wandb.name = name + "/train"
            config.wandb.group = f"fewstep-{updates}"
            inference = tuple(
                InferenceJob(
                    overrides={
                        "sampler.steps": nfe - 1,
                        "evaluation.max_examples": examples,
                        "evaluation.policy": f"nfe-{nfe}",
                        "wandb.name": name + f"/nfe-{nfe}",
                    }
                )
                for nfe in budgets
            )
            comparisons.append(
                replace(job, name=name, config=config, inference=inference)
            )
    return sorted(comparisons, key=lambda job: "-base-" not in job.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path.home() / ".cache/duo/batteries/three-tasks-20m-bf16-20261012",
    )
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    jobs = build_comparison(
        discover(args.base_root), args.updates, args.batch_size, args.examples
    )
    from dataclasses import asdict
    import json

    rows = []
    for job in jobs:
        row = asdict(job)
        row["config"] = OmegaConf.to_container(job.config, resolve=True)
        rows.append(row)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        f"{len(jobs)} jobs (including six zero-update bases), {sum(len(j.inference) for j in jobs)} evaluations"
    )
    if args.submit:
        print(f"Submitted {SubmissionQueue(args.queue).submit(jobs)} jobs")


if __name__ == "__main__":
    main()
