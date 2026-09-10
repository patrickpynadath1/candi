#!/usr/bin/env python3
"""Manifest-driven rank ablations. No full runs without --execute.

Examples:
  python new_scripts/sudoku_rank_ablation.py plan
  python new_scripts/sudoku_rank_ablation.py smoke --difficulties easy --seeds 1
  python new_scripts/sudoku_rank_ablation.py train --gpus 0 1 --execute
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def entries(manifest, args):
    variants = args.variants or list(manifest.variants)
    for variant in variants:
        if variant not in manifest.variants:
            raise ValueError(f"unknown variant: {variant}")
    jobs = []
    for v in variants:
        allowed = manifest.variants[v].get("modes", manifest.modes)
        modes = [m for m in (args.modes or manifest.modes) if m in allowed]
        jobs.extend(dict(variant=v, mode=m, difficulty=d, seed=s)
                    for m, d, s in itertools.product(modes,
                        args.difficulties or manifest.difficulties,
                        args.seeds or manifest.seeds))
    if not jobs:
        raise ValueError("no configured variants match the requested modes")
    return jobs



def overrides(manifest, job, root, smoke=False):
    entry = manifest.variants[job["variant"]]
    directory = root / job["variant"] / job["mode"] / job["difficulty"] / f"seed_{job['seed']}"
    values = [
        f"+experiment={entry.experiment}", *entry.overrides,
        "+callbacks=reasoning_eval",
        f"algo.pure_continuous={str(job['mode'] == 'pure').lower()}",
        f"data.difficulty={job['difficulty']}", f"seed={job['seed']}",
        f"data.cache_dir={root / 'data'}", f"checkpointing.save_dir={directory}",
        f"hydra.run.dir={directory}", "trainer.devices=1", "trainer.num_nodes=1",
        "wandb=null", "callbacks.checkpoint_every_n_steps.every_n_train_steps=5000",
    ]
    if smoke:
        values += [
            "callbacks.reasoning_eval.every_n_train_steps=1",
            "callbacks.reasoning_eval.subset_size=4",
            "callbacks.reasoning_eval.batch_size=2",
            "callbacks.reasoning_eval.num_steps=4",
            "strategy=single_cpu",
            "model.hidden_size=32", "model.cond_dim=32", "model.n_blocks=1",
            "model.n_heads=2", "model.dropout=0", "trainer.accelerator=cpu",
            "trainer.precision=32-true", "trainer.max_steps=2",
            "trainer.accumulate_grad_batches=1", "trainer.val_check_interval=2",
            "trainer.num_sanity_val_steps=0", "loader.global_batch_size=4",
            "loader.eval_global_batch_size=4", "loader.num_workers=0",
            "data.num_train=8", "data.num_valid=4", "training.ema=0",
            "noise.adaptive_warmup_steps=0", "noise.adaptive_buffer_size=4",
            "noise.adaptive_refit_every=1", "noise.adaptive_n_knots=3",
            "callbacks.checkpoint_every_n_steps.every_n_train_steps=1",
        ]
    return directory, values


def call_main(values, directory, stage, env):
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "main.py"), *values]
    (directory / f"{stage}.command.json").write_text(json.dumps(command, indent=2))
    started = time.monotonic()
    with (directory / f"{stage}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = dict(returncode=result.returncode, seconds=time.monotonic()-started)
    (directory / f"{stage}.status.json").write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError(f"{stage} failed: {directory / (stage + '.log')}")


def run_job(manifest, job, args, root, gpu):
    smoke = args.command == "smoke"
    directory, values = overrides(manifest, job, root, smoke)
    metadata = dict(job, experiment=manifest.variants[job["variant"]].experiment,
                    overrides=values, smoke=smoke)
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "run.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError(f"configuration differs from existing run: {directory}; use a new --root")
    metadata_path.write_text(json.dumps(dict(metadata, fingerprint=fingerprint), indent=2))
    env = dict(os.environ, OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env["CUDA_VISIBLE_DEVICES"] = "" if smoke else str(gpu)
    print(f"{args.command}: {job}", flush=True)
    if args.command in {"train", "smoke"}:
        done = directory / "train.complete"
        if not done.exists():
            call_main(values, directory, "train", env)
            done.touch()
    if args.command in {"eval", "smoke"}:
        checkpoint = directory / "checkpoints/last.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for nfe in ([4] if smoke else manifest.nfe):
            evaluation = directory / "eval" / f"nfe_{nfe}"
            if (evaluation / "eval.complete").exists():
                continue
            eval_values = values + [
                "mode=sudoku_eval", f"eval.checkpoint_path={checkpoint}",
                f"sampling.steps={nfe}", f"sudoku.output_dir={evaluation}",
                f"hydra.run.dir={evaluation}", "sudoku.batch_size=4" if smoke else "sudoku.batch_size=64",
            ]
            if smoke:
                eval_values.append("eval.disable_ema=true")
            call_main(eval_values, evaluation, "eval", env)
            (evaluation / "eval.complete").touch()


def aggregate(root):
    rows = []
    for metadata_path in sorted(root.glob("*/*/*/seed_*/run.json")):
        metadata = json.loads(metadata_path.read_text())
        for result_path in sorted(metadata_path.parent.glob("eval/nfe_*/results.json")):
            result = json.loads(result_path.read_text())
            rows.append({
                **{k: metadata[k] for k in ("variant", "mode", "difficulty", "seed", "smoke")},
                "requested_nfe": int(result_path.parent.name.split("_")[-1]),
                **{k: v for k, v in result.items() if k != "records"},
            })
    if not rows:
        raise ValueError(f"no evaluation results in {root}")
    destination = root / "summary.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with destination.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(destination)
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in ("variant", "mode", "difficulty", "requested_nfe", "smoke"))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, group in grouped.items():
        summary = dict(zip(("variant", "mode", "difficulty", "requested_nfe", "smoke"), key))
        summary["num_seeds"] = len(group)
        for metric in ("accuracy", "cell_accuracy", "digit_accuracy", "valid_grid_rate",
                       "valid_solution_rate", "clue_consistency_rate", "mean_nfe", "seconds_per_puzzle"):
            values = [row[metric] for row in group if row.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(summary)
    with (root / "summary_by_variant.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in summaries for k in row)))
        writer.writeheader()
        writer.writerows(summaries)


def prepare(jobs, root, smoke=False):
    # Build each shared cache once, before concurrent training workers start.
    for difficulty in dict.fromkeys(j["difficulty"] for j in jobs):
        command = [
            sys.executable, str(ROOT / "new_scripts/prepare_sudoku.py"),
            "--difficulty", difficulty, "--cache-dir", str(root / "data"),
        ]
        if smoke:
            command += ["--num-train", "8", "--num-valid", "4", "--workers", "1"]
        subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "smoke", "train", "eval", "aggregate", "prepare"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/sweeps/sudoku_rank.yaml")
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/sudoku_rank")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=["pure", "hybrid"])
    parser.add_argument("--difficulties", nargs="+", choices=["easy", "medium", "hard"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--smoke-workers", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = OmegaConf.load(args.manifest)
    root = args.root.resolve()
    if args.command == "smoke":
        if args.smoke_workers < 1:
            parser.error("--smoke-workers must be positive")
        args.gpus = list(range(args.smoke_workers))
        root = root / "smoke"
        if args.seeds is None:
            args.seeds = [1]
    if args.command == "aggregate":
        aggregate(root)
        return
    jobs = entries(manifest, args)
    if args.command == "plan" or (args.command in {"train", "eval", "prepare"} and not args.execute):
        print(json.dumps(dict(num_runs=len(jobs), nfe=list(manifest.nfe), jobs=[
            dict(job, overrides=overrides(manifest, job, root)[1]) for job in jobs
        ]), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(jobs, root)
        return
    if args.command in {"train", "smoke"}:
        prepare(jobs, root, args.command == "smoke")
    # Sequential per-GPU queues; failed jobs remain resumable and are reported.
    def queue(index):
        failures = []
        for job in jobs[index::len(args.gpus)]:
            try:
                run_job(manifest, job, args, root, args.gpus[index])
            except (RuntimeError, ValueError, FileNotFoundError) as error:
                print(str(error), file=sys.stderr, flush=True)
                failures.append(str(error))
        return failures
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        failures = list(itertools.chain.from_iterable(pool.map(queue, range(len(args.gpus)))))
    if failures:
        raise SystemExit(f"{len(failures)} failed jobs; see logs above")
    if args.command in {"eval", "smoke"}:
        aggregate(root)


if __name__ == "__main__":
    main()
