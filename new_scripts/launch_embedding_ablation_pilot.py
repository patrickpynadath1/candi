#!/usr/bin/env python3
"""Durable 5K Sudoku Hard pilot for the remaining embedding ablations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from omegaconf import OmegaConf

from sudoku_rank_ablation import ROOT, overrides as rank_overrides


PILOTS = [
    ("embedding_linear_adaptive", "pure", "gaussian_adaptive"),
    ("embedding_linear_rank", "pure", "gaussian_rank_linear"),
    ("embedding_geometric_rank", "pure", "gaussian_rank_geometric"),
    ("embedding_linear", "hybrid", "gaussian_linear"),
    ("embedding_linear_rank", "hybrid", "gaussian_rank_linear"),
    ("embedding_geometric_rank", "hybrid", "gaussian_rank_geometric"),
    (
        "embedding_geometric_rank_adaptive",
        "hybrid",
        "gaussian_rank_geometric_adaptive",
    ),
    (
        "slerp_geometric_rank_adaptive",
        "hybrid",
        "slerp_rank_geometric_adaptive",
    ),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def snapshot(root):
    destination = root / "source"
    if destination.exists():
        return destination
    destination.mkdir(parents=True)
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, destination / path.name)
    for name in (
        "algos", "models", "samplers", "tasks", "configs", "integral",
        "new_scripts",
    ):
        shutil.copytree(
            ROOT / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    hashes = {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.rglob("*")
        if path.is_file()
    }
    write_json(root / "source_hashes.json", hashes)
    return destination


def make_job(matrix, root, variant, mode, label, seed, project, entity):
    base_job = {
        "variant": variant,
        "mode": mode,
        "difficulty": "hard",
        "seed": seed,
    }
    _, values = rank_overrides(matrix, base_job, root)
    values = [
        value
        for value in values
        if not value.startswith(
            (
                "wandb=", "trainer.max_steps=", "loader.num_workers=",
                "checkpointing.save_dir=", "hydra.run.dir=",
            )
        )
    ]
    directory = root / "runs" / f"{mode}_{label}" / f"seed_{seed}"
    identity = f"{root.name}:{mode}:{label}:{seed}"
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    run_name = f"embedding_{mode}_{label}_sudoku_hard_seed_{seed}"
    values += [
        "+algo.input_embedding_dim=512",
        "+algo.output_embedding_dim=512",
        "trainer.max_steps=5000",
        "loader.num_workers=4",
        "callbacks.reasoning_eval.every_n_train_steps=1000",
        "callbacks.checkpoint_every_n_steps.every_n_train_steps=5000",
        "callbacks.checkpoint_every_n_steps.save_top_k=0",
        "callbacks.checkpoint_every_n_steps.save_last=true",
        "callbacks.checkpoint_monitor.save_top_k=0",
        "callbacks.checkpoint_monitor.monitor=null",
        "checkpointing.resume_from_ckpt=false",
        f"checkpointing.save_dir={directory}",
        f"hydra.run.dir={directory}",
        f"wandb.project={project}",
        f"+wandb.entity={entity}",
        f"wandb.id={run_id}",
        f"wandb.name={run_name}",
        "wandb.group=embedding_ablation_5k_sudoku_hard",
        "wandb.job_type=train",
        f"wandb.tags=[embedding,{mode},{label},sudoku_hard,pilot_5k]",
    ]
    return {
        **base_job,
        "label": label,
        "embedding_dimension": 512,
        "directory": str(directory),
        "run_id": run_id,
        "run_name": run_name,
        "wandb_url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        "overrides": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpus", nargs="+", default=["0", "2", "3", "4", "5", "6"])
    parser.add_argument("--project", default="reasoning-embedding-pure-ablations")
    parser.add_argument("--entity", default="rz-text-dlp")
    parser.add_argument(
        "--hard-cache",
        type=Path,
        default=Path(
            "/home/patrick/.cache/duo/data_cache/"
            "sudoku_hard_train48000_valid2000_seed42"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    if args.detach:
        if not args.execute:
            parser.error("--detach requires --execute")
        root.mkdir(parents=True, exist_ok=True)
        command = [argument for argument in sys.argv if argument != "--detach"]
        with (root / "supervisor.log").open("a") as stream:
            process = subprocess.Popen(
                [sys.executable, *command], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(process.pid)
        return

    matrix = OmegaConf.load(ROOT / "configs/sweeps/sudoku_rank.yaml")
    jobs = [
        make_job(
            matrix, root, variant, mode, label, args.seed, args.project, args.entity
        )
        for variant, mode, label in PILOTS
    ]
    plan = {
        "task": "sudoku_hard",
        "embedding_dimension": 512,
        "max_steps": 5000,
        "evaluation_every_steps": 1000,
        "seed": args.seed,
        "gpus": args.gpus,
        "project": args.project,
        "jobs": jobs,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return

    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign_path = root / "campaign.json"
    if campaign_path.exists() and json.loads(campaign_path.read_text()) != plan:
        raise ValueError("existing campaign differs; use a new root")
    write_json(campaign_path, plan)
    source = snapshot(root)

    data = root / "data"
    data.mkdir(exist_ok=True)
    cache_link = data / args.hard_cache.name
    if not cache_link.exists():
        cache_link.symlink_to(args.hard_cache)
    if not cache_link.exists():
        raise FileNotFoundError(args.hard_cache)

    states = {str(index): "queued" for index in range(len(jobs))}
    mutex = threading.Lock()

    def save():
        write_json(
            root / "status.json",
            {
                "pid": os.getpid(),
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "jobs": states,
            },
        )

    def worker(gpu):
        while True:
            with mutex:
                queued = [i for i in range(len(jobs)) if states[str(i)] == "queued"]
                if not queued:
                    return
                index = queued[0]
                states[str(index)] = f"waiting for GPU {gpu}"
                save()
            job = jobs[index]
            directory = Path(job["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            while True:
                usage = subprocess.check_output(
                    [
                        "nvidia-smi", "-i", str(gpu),
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                memory, utilization = map(int, usage.strip().split(","))
                if memory < 1024 and utilization < 10:
                    break
                time.sleep(30)
            if shutil.disk_usage(root).free < 12 * 2**30:
                raise RuntimeError("less than 12 GiB free disk")
            command = [sys.executable, str(source / "main.py"), *job["overrides"]]
            write_json(directory / "run.json", {**job, "gpu": gpu, "command": command})
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                TOKENIZERS_PARALLELISM="false", WANDB_MODE="online",
                WANDB_RESUME="allow", PYTHONUNBUFFERED="1",
            )
            started = time.monotonic()
            with (directory / "train.log").open("a") as stream:
                process = subprocess.Popen(
                    command, cwd=source, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT,
                )
                with mutex:
                    states[str(index)] = f"running on GPU {gpu}, pid {process.pid}"
                    save()
                returncode = process.wait()
            write_json(
                directory / "train.status.json",
                {"returncode": returncode, "seconds": time.monotonic() - started},
            )
            with mutex:
                states[str(index)] = (
                    "complete" if returncode == 0 else f"failed: training exit {returncode}"
                )
                save()

    with mutex:
        save()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    print(json.dumps(states, indent=2))


if __name__ == "__main__":
    main()
