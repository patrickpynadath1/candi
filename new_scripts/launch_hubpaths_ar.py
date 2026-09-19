#!/usr/bin/env python3
"""Launch one durable, W&B-logged HubPaths AR run with Sudoku's backbone size."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def snapshot(directory):
    dest = directory / "source"
    dest.mkdir()
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, dest / path.name)
    for name in ("algos", "models", "samplers", "tasks", "configs", "integral"):
        shutil.copytree(
            ROOT / name,
            dest / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    write_json(
        directory / "source_hashes.json",
        {
            str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.rglob("*")
            if p.is_file()
        },
    )
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--entity", default="rz-text-dlp")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    directory, dataset = args.output.resolve(), args.dataset.resolve()
    metadata = json.loads((dataset / "metadata.json").read_text())
    if metadata["config"]["n_hints"]:
        raise ValueError("AR needs a dataset with n_hints=0")
    directory.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(directory),
            "--dataset",
            str(dataset),
            "--gpu",
            str(args.gpu),
            "--entity",
            args.entity,
        ]
        with (directory / "supervisor.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"supervisor_pid": process.pid, "directory": str(directory)}))
        return
    with (directory / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "run.json").exists():
            raise ValueError("run already exists; choose another output directory")
        source = snapshot(directory)
        run_id = uuid.uuid4().hex[:12]
        overrides = [
            "+experiment=hubpaths_ar_sudoku_size",
            f"data.prepared_dir={dataset}",
            f"model.length={metadata['seq_len']}",
            "strategy=single_gpu",
            "trainer.devices=1",
            "trainer.num_nodes=1",
            "trainer.precision=bf16-mixed",
            "+trainer.enable_progress_bar=false",
            "trainer.num_sanity_val_steps=0",
            "trainer.log_every_n_steps=10",
            "seed=1",
            "checkpointing.resume_from_ckpt=false",
            f"checkpointing.save_dir={directory}",
            f"hydra.run.dir={directory}",
            f"callbacks.hubpaths_eval.output_dir={directory / 'eval'}",
            f"+wandb.entity={args.entity}",
            f"wandb.id={run_id}",
            f"+wandb.save_dir={directory}",
            "+wandb.mode=online",
            "wandb.job_type=train",
            "wandb.tags=[hubpaths,ar,sudoku-sized,10000-nodes,3-waypoints,100k-examples,20k-steps,seed1]",
        ]
        command = [sys.executable, str(source / "main.py"), *overrides]
        manifest = {
            "run_id": run_id,
            "wandb_url": f"https://wandb.ai/{args.entity}/hubpaths/runs/{run_id}",
            "dataset": str(dataset),
            "dataset_metadata": metadata,
            "command": command,
            "gpu": args.gpu,
            "source": str(source),
            "supervisor_pid": os.getpid(),
            "max_steps": 20000,
            "model": {"hidden_size": 512, "n_blocks": 8, "n_heads": 8, "cond_dim": 128},
        }
        write_json(directory / "run.json", manifest)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(args.gpu),
            OMP_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            WANDB_MODE="online",
            PYTHONUNBUFFERED="1",
        )
        started = time.time()
        with (directory / "train.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write_json(
                directory / "status.json",
                {
                    "state": "running",
                    "pid": process.pid,
                    "started_unix": started,
                    "wandb_url": manifest["wandb_url"],
                },
            )
            code = process.wait()
        write_json(
            directory / "status.json",
            {
                "state": "complete" if code == 0 else "failed",
                "pid": process.pid,
                "returncode": code,
                "seconds": time.time() - started,
                "wandb_url": manifest["wandb_url"],
            },
        )
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
