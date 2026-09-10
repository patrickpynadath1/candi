#!/usr/bin/env python3
"""Run a durable 5K adaptive-SLERP embedding-dimension sweep on Sudoku Hard."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMS = [8, 16, 32, 64, 128, 256, 512]


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
    for name in ("algos", "models", "samplers", "tasks", "configs", "integral"):
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


def job_spec(root, dimension, seed, project, entity):
    directory = root / "runs" / f"embed_dim_{dimension:03d}" / f"seed_{seed}"
    identity = f"{root.name}:slerp-adaptive:{dimension}:{seed}"
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    run_name = f"slerp_adaptive_embed_dim_{dimension:03d}_sudoku_hard_seed_{seed}"
    values = [
        "+experiment=sudoku_sfm_paper_hard",
        f"model.input_embedding_dim={dimension}",
        f"seed={seed}",
        "data.cache_dir=" + str(root / "data"),
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.max_steps=5000",
        "loader.num_workers=4",
        "callbacks.sudoku_eval.every_n_train_steps=1000",
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
        "wandb.group=slerp_adaptive_embedding_dimension",
        "wandb.job_type=train",
        f"wandb.tags=[slerp,adaptive,embedding_dimension_{dimension},sudoku_hard]",
    ]
    return {
        "dimension": dimension,
        "seed": seed,
        "directory": str(directory),
        "run_id": run_id,
        "run_name": run_name,
        "wandb_url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        "overrides": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=DEFAULT_DIMS)
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
    if any(d < 2 for d in args.dimensions) or len(set(args.dimensions)) != len(
        args.dimensions
    ):
        parser.error("dimensions must be unique integers of at least two")

    root = args.root.resolve()
    if args.detach:
        if not args.execute:
            parser.error("--detach requires --execute")
        root.mkdir(parents=True, exist_ok=True)
        command = [arg for arg in sys.argv if arg != "--detach"]
        with (root / "supervisor.log").open("a") as stream:
            process = subprocess.Popen(
                [sys.executable, *command],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(process.pid)
        return
    jobs = [
        job_spec(root, dimension, args.seed, args.project, args.entity)
        for dimension in args.dimensions
    ]
    plan = {
        "task": "sudoku_hard",
        "path": "truncated_slerp",
        "adaptive_schedule": True,
        "auto_truncation": True,
        "max_steps": 5000,
        "evaluation_every_steps": 1000,
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
    plan_path = root / "campaign.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("existing campaign differs; use a new root")
    write_json(plan_path, plan)
    source = snapshot(root)

    data = root / "data"
    data.mkdir(exist_ok=True)
    cache_link = data / args.hard_cache.name
    if not cache_link.exists():
        cache_link.symlink_to(args.hard_cache)
    if not cache_link.exists():
        raise FileNotFoundError(args.hard_cache)

    states = {str(i): "queued" for i in range(len(jobs))}
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
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="1",
                TOKENIZERS_PARALLELISM="false",
                WANDB_MODE="online",
                WANDB_RESUME="allow",
                PYTHONUNBUFFERED="1",
            )
            started = time.monotonic()
            with (directory / "train.log").open("a") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
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
