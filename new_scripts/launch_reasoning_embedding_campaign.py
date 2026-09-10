#!/usr/bin/env python3
"""Durable local GPU queue for the pure embedding ablations; launch with --execute."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from omegaconf import OmegaConf
from sudoku_rank_ablation import ROOT, overrides as sudoku_overrides
from reasoning_ablation import overrides as task_overrides

VARIANTS = [
    "embedding_linear_adaptive",
    "embedding_geometric_rank",
    "embedding_linear_rank",
    "slerp_truncated_adaptive",
    "slerp_geometric_rank",
]
TASKS = ["sudoku_hard", "sudoku_extreme", "zebra_4x4", "maze_unique"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def values_for(job, root, prepared, project, entity, wandb_suffix=None):
    matrix = OmegaConf.load(ROOT / "configs/sweeps/sudoku_rank.yaml")
    profiles = OmegaConf.load(ROOT / "configs/sweeps/reasoning_tasks.yaml").profiles
    if job["profile"] == "sudoku_hard":
        _, values = sudoku_overrides(matrix, dict(job, difficulty="hard"), root)
    else:
        _, values = task_overrides(matrix, profiles, job, root, prepared)
    directory = root / "runs" / job["profile"] / job["variant"] / f"seed_{job['seed']}"
    values = [
        v
        for v in values
        if not v.startswith(("wandb=", "checkpointing.save_dir=", "hydra.run.dir="))
    ]
    run_id = hashlib.sha256(f"{root.name}:{job}".encode()).hexdigest()[:16]
    run_name = f"{job['profile']}_{job['variant']}_seed_{job['seed']}"
    if wandb_suffix:
        run_name += f"_{wandb_suffix}"
    values += [
        f"checkpointing.save_dir={directory}",
        f"hydra.run.dir={directory}",
        f"wandb.project={project}",
        f"+wandb.entity={entity}",
        f"wandb.id={run_id}",
        f"wandb.name={run_name}",
        f"wandb.group={job['profile']}",
        "wandb.job_type=train",
        f"wandb.tags=[pure,embedding,{job['variant']},{job['profile']}]",
        "callbacks.checkpoint_every_n_steps.save_top_k=0",
        "callbacks.checkpoint_every_n_steps.save_last=true",
        "callbacks.checkpoint_monitor.save_top_k=0",
        "callbacks.checkpoint_monitor.monitor=null",
    ]
    if job["profile"] in ("sudoku_extreme", "maze_unique"):
        values += ["data.full_training=true", "trainer.limit_val_batches=0"]
    return directory, values, run_id


def snapshot(root):
    destination = root / "source"
    if destination.exists():
        return destination
    destination.mkdir(parents=True)
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, destination / path.name)
    for name in (
        "algos",
        "models",
        "samplers",
        "tasks",
        "configs",
        "integral",
        "new_scripts",
    ):
        shutil.copytree(
            ROOT / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    hashes = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file()
    }
    write_json(root / "source_hashes.json", hashes)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data_cache/reasoning")
    parser.add_argument(
        "--hard-cache",
        type=Path,
        default=Path(
            "/home/patrick/.cache/duo/data_cache/sudoku_hard_train48000_valid2000_seed42"
        ),
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "2", "3", "4", "5", "6"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--project", default="reasoning-embedding-pure-ablations")
    parser.add_argument("--entity", default="rz-text-dlp")
    parser.add_argument("--wandb-suffix")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    prepared = args.prepared.resolve()
    selected_variants = args.variants or VARIANTS
    selected_tasks = args.tasks or TASKS
    jobs = [
        dict(profile=p, variant=v, mode="pure", seed=s)
        for s, p, v in itertools.product(args.seeds, selected_tasks, selected_variants)
    ]
    plan = dict(
        project=args.project,
        entity=args.entity,
        gpus=args.gpus,
        jobs=[
            dict(
                j,
                overrides=values_for(
                    j, root, prepared, args.project, args.entity, args.wandb_suffix
                )[1],
            )
            for j in jobs
        ],
        max_steps=20000,
        monitoring=dict(interval=1000, puzzles=128, nfe=45),
        final_checkpoint_only=True,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "campaign.json").exists() and json.loads(
        (root / "campaign.json").read_text()
    ) != plan:
        raise ValueError("existing campaign differs; use a new root")
    write_json(root / "campaign.json", plan)
    source = snapshot(root)
    cache = root / "data" / args.hard_cache.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.symlink_to(args.hard_cache)
    if not cache.exists():
        raise FileNotFoundError(args.hard_cache)
    statuses = {p: "preparing" for p in selected_tasks}
    if "sudoku_hard" in statuses:
        statuses["sudoku_hard"] = "ready"
    states = {str(i): "queued" for i in range(len(jobs))}
    mutex = threading.Lock()

    def save():
        write_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                datasets=statuses,
                jobs=states,
            ),
        )

    def prepare_task(name):
        try:
            path = prepared / name
            if not (path / "manifest.json").exists():
                command = [
                    sys.executable,
                    str(source / "new_scripts/prepare_reasoning.py"),
                    "--task",
                    "zebra" if name == "zebra_4x4" else name,
                    "--raw",
                    str(ROOT / "data_cache/reasoning_raw"),
                    "--output",
                    str(path),
                ]
                if name == "zebra_4x4":
                    command += ["--layout", "4", "4"]
                else:
                    command += ["--full-training"]
                with (root / f"prepare_{name}.log").open("a") as stream:
                    subprocess.run(
                        command,
                        cwd=source,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest["source_limit"] is not None or manifest["full_training"] != (
                name != "zebra_4x4"
            ):
                raise ValueError("prepared release profile mismatch")
            if name == "zebra_4x4" and manifest["layout"] != [4, 4]:
                raise ValueError("Zebra layout mismatch")
            with mutex:
                statuses[name] = "ready"
                save()
        except Exception as error:
            with mutex:
                statuses[name] = f"failed: {error}"
                save()

    def worker(gpu):
        while True:
            with mutex:
                for i, j in enumerate(jobs):
                    if states[str(i)] == "queued" and statuses[j["profile"]].startswith(
                        "failed"
                    ):
                        states[str(i)] = "blocked: data preparation failed"
                candidates = [
                    i
                    for i, j in enumerate(jobs)
                    if states[str(i)] == "queued" and statuses[j["profile"]] == "ready"
                ]
                if not candidates:
                    if not any(v == "queued" for v in states.values()):
                        return
                    index = None
                else:
                    index = candidates[0]
                    states[str(index)] = f"starting on GPU {gpu}"
                    save()
            if index is None:
                time.sleep(5)
                continue
            job = jobs[index]
            directory, values, run_id = values_for(
                job, root, prepared, args.project, args.entity, args.wandb_suffix
            )
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "train.complete").exists():
                with mutex:
                    states[str(index)] = "complete"
                    save()
                continue
            try:
                # Avoid starting over another user's workload if GPU occupancy changes.
                while True:
                    usage = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            str(gpu),
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    )
                    memory, utilization = map(int, usage.strip().split(","))
                    if memory < 1024 and utilization < 10:
                        break
                    with mutex:
                        states[str(index)] = f"waiting for GPU {gpu}"
                        save()
                    time.sleep(30)
                if shutil.disk_usage(root).free < 8 * 2**30:
                    raise RuntimeError("less than 8 GiB free disk")
                command = [sys.executable, str(source / "main.py"), *values]
                write_json(
                    directory / "run.json",
                    dict(
                        job=job,
                        wandb_id=run_id,
                        wandb_project=args.project,
                        wandb_url=f"https://wandb.ai/{args.entity}/{args.project}/runs/{run_id}",
                        gpu=gpu,
                        command=command,
                    ),
                )
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS="1",
                    TOKENIZERS_PARALLELISM="false",
                    WANDB_MODE="online",
                    WANDB_RESUME="allow",
                )
                started = time.monotonic()
                with (directory / "train.log").open("a") as stream:
                    process = subprocess.Popen(
                        command,
                        cwd=source,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    with mutex:
                        states[str(index)] = f"running on GPU {gpu}, pid {process.pid}"
                        save()
                    code = process.wait()
                write_json(
                    directory / "train.status.json",
                    dict(returncode=code, seconds=time.monotonic() - started),
                )
                if code:
                    raise RuntimeError(f"training exit {code}: {directory}/train.log")
                (directory / "train.complete").touch()
                with mutex:
                    states[str(index)] = "complete"
                    save()
            except Exception as error:
                with mutex:
                    states[str(index)] = f"failed: {error}"
                    save()

    with mutex:
        save()
    with (
        ThreadPoolExecutor(max_workers=3) as preparation,
        ThreadPoolExecutor(max_workers=len(args.gpus)) as execution,
    ):
        prep = [
            preparation.submit(prepare_task, p)
            for p in selected_tasks
            if p != "sudoku_hard"
        ]
        runs = [execution.submit(worker, g) for g in args.gpus]
        for future in prep + runs:
            future.result()
    print(json.dumps(states, indent=2), flush=True)


if __name__ == "__main__":
    main()
