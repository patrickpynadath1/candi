#!/usr/bin/env python3
"""Evaluate every trained Sudoku method, using EB for every hybrid method."""

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def sampler_for(job):
    experiment = job["method"]["experiment"]
    if job["method"]["mode"] == "hybrid":
        if experiment == "sudoku_candi_rank":
            return "candi_entropy_bounded"
        return "candi_embedding_entropy_bounded"
    if experiment == "sudoku_sfm_truncated_adaptive":
        return "sfm"
    if experiment == "sudoku_candi_rank":
        return "candi"
    return "candi_embedding"


def evaluate(job, gpu, output_root, steps, max_examples, gamma, batch_size):
    name = job["method_name"]
    output = output_root / name
    result_path = output / "results.json"
    if result_path.exists():
        return name, "complete", gpu
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(job["directory"]) / "checkpoints" / "last.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    sampler = sampler_for(job)
    overrides = list(job["overrides"]) + [
        "mode=sudoku_eval",
        f"sampling.predictor={sampler}",
        f"sampling.steps={steps}",
        f"sampling.entropy_bound={gamma}",
        "sampling.greedy_clean_estimate=true",
        f"sudoku.max_examples={max_examples}",
        f"sudoku.batch_size={batch_size}",
        f"loader.eval_global_batch_size={batch_size}",
        "loader.num_workers=1",
        f"eval.checkpoint_path={checkpoint}",
        f"sudoku.output_dir={output}",
        f"hydra.run.dir={output}",
        "wandb=null",
    ]
    metadata = {
        "method": job["method"],
        "method_name": name,
        "checkpoint": str(checkpoint),
        "sampler": sampler,
        "steps": steps,
        "max_examples": max_examples,
        "entropy_bound": gamma if job["method"]["mode"] == "hybrid" else None,
        "greedy_clean_estimate": (
            True if job["method"]["mode"] == "hybrid" else None
        ),
        "gpu": str(gpu),
    }
    (output / "evaluation.json").write_text(json.dumps(metadata, indent=2))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    started = time.time()
    with (output / "eval.log").open("w") as log:
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "main.py"),
                "--config-name=config",
                *overrides,
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    metadata["wall_seconds"] = time.time() - started
    (output / "evaluation.json").write_text(json.dumps(metadata, indent=2))
    return name, "evaluated", gpu


def summarize(output_root):
    rows = []
    for result_path in sorted(output_root.glob("*/results.json")):
        result = json.loads(result_path.read_text())
        metadata = json.loads((result_path.parent / "evaluation.json").read_text())
        rows.append({
            "method": metadata["method_name"],
            "label": metadata["method"]["label"],
            "mode": metadata["method"]["mode"],
            "sampler": metadata["sampler"],
            "steps": metadata["steps"],
            "entropy_bound": metadata["entropy_bound"],
            "accuracy": result["accuracy"],
            "cell_accuracy": result["cell_accuracy"],
            "valid_grid_rate": result["valid_grid_rate"],
            "clue_consistency_rate": result["clue_consistency_rate"],
            "mean_nfe": result["mean_nfe"],
            "num_correct": result["num_correct"],
            "num_total": result["num_total"],
            "evaluation_seconds": result["evaluation_seconds"],
        })
    if rows:
        with (output_root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--entropy-bound", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    args.output = args.output.resolve()

    campaign = json.loads(args.campaign.read_text())
    jobs = [job for job in campaign["jobs"] if job["task"] == "sudoku_hard"]
    if not jobs:
        raise ValueError("campaign contains no Sudoku Hard jobs")
    if args.methods:
        requested = set(args.methods)
        jobs = [job for job in jobs if job["method_name"] in requested]
        missing = requested - {job["method_name"] for job in jobs}
        if missing:
            raise ValueError(f"unknown methods: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "campaign": str(args.campaign.resolve()),
        "methods": [job["method_name"] for job in jobs],
        "gpus": args.gpus,
        "steps": args.steps,
        "max_examples": args.max_examples,
        "entropy_bound": args.entropy_bound,
        "hybrid_sampler_policy": "entropy_bounded_only",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))

    def gpu_queue(gpu, assigned):
        completed = []
        for job in assigned:
            result = evaluate(
                job,
                gpu,
                args.output,
                args.steps,
                args.max_examples,
                args.entropy_bound,
                args.batch_size,
            )
            print(result, flush=True)
            completed.append(result)
        return completed

    queues = {
        gpu: jobs[index::len(args.gpus)] for index, gpu in enumerate(args.gpus)
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(gpu_queue, gpu, assigned)
            for gpu, assigned in queues.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    rows = summarize(args.output)
    if len(rows) != len(jobs):
        raise RuntimeError(f"only {len(rows)} of {len(jobs)} evaluations completed")
    print(args.output / "summary.csv")


if __name__ == "__main__":
    main()
