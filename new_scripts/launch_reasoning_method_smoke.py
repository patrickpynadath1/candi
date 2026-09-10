#!/usr/bin/env python3
"""Durable 5K cross-task validation campaign for five reasoning methods."""

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

from sudoku_rank_ablation import ROOT


METHODS = {
    "slerp_adaptive": {
        "experiment": "sudoku_sfm_truncated_adaptive",
        "mode": "pure",
        "overrides": [],
        "tasks": ["zebra_4x4", "sudoku_extreme", "maze_unique"],
    },
    "embedding_gaussian_adaptive": {
        "variant": "embedding_linear_adaptive",
        "mode": "pure",
        "tasks": ["zebra_4x4", "sudoku_extreme", "maze_unique"],
    },
    "hybrid_slerp_rank_geo_adaptive": {
        "variant": "slerp_geometric_rank_adaptive",
        "mode": "hybrid",
        "tasks": ["zebra_4x4", "sudoku_hard", "sudoku_extreme", "maze_unique"],
    },
    "hybrid_embedding_rank_geo_adaptive": {
        "variant": "embedding_geometric_rank_adaptive",
        "mode": "hybrid",
        "tasks": ["zebra_4x4", "sudoku_hard", "sudoku_extreme", "maze_unique"],
    },
    "hybrid_onehot_rank_geo_adaptive": {
        "variant": "candi_geometric_adaptive",
        "mode": "hybrid",
        "tasks": ["zebra_4x4", "sudoku_hard", "sudoku_extreme", "maze_unique"],
    },
}


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


def task_overrides(profile_name, profile, root, prepared):
    if profile_name == "sudoku_hard":
        return [
            "data=sudoku",
            "data.difficulty=hard",
            f"data.cache_dir={root / 'data'}",
            "model.length=180",
            "loader.global_batch_size=256",
            "loader.eval_global_batch_size=128",
            "reasoning.batch_size=64",
        ]
    values = [
        f"data={profile.task}",
        f"data.prepared_dir={prepared / profile_name}",
        f"model.length={profile.length}",
        f"data.prompt_length={profile.prompt_length}",
        f"loader.global_batch_size={profile.batch_size}",
        f"loader.eval_global_batch_size={profile.batch_size}",
        f"reasoning.batch_size={profile.batch_size}",
    ]
    if profile.get("full_training", False):
        values += ["data.full_training=true", "trainer.limit_val_batches=0"]
    return values


def make_job(matrix, profiles, root, prepared, method_name, task, spec, seed,
             project, entity, hybrid_loss_weighting):
    if "variant" in spec:
        entry = matrix.variants[spec["variant"]]
        experiment = entry.experiment
        method_values = list(entry.overrides)
    else:
        experiment = spec["experiment"]
        method_values = list(spec["overrides"])
    mode = spec["mode"]
    directory = root / "runs" / method_name / task / f"seed_{seed}"
    loss_suffix = (
        f"_loss_{hybrid_loss_weighting}" if mode == "hybrid" else ""
    )
    run_name = f"smoke_{method_name}_{task}{loss_suffix}_seed_{seed}"
    identity = (
        f"{root.name}:{method_name}:{task}:{seed}:{hybrid_loss_weighting}"
    )
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    values = [
        f"+experiment={experiment}",
        *method_values,
        "+callbacks=reasoning_eval",
        f"seed={seed}",
        *task_overrides(task, profiles.get(task), root, prepared),
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.max_steps=5000",
        "loader.num_workers=4",
        "callbacks.reasoning_eval.every_n_train_steps=1000",
        "callbacks.reasoning_eval.subset_size=128",
        "callbacks.reasoning_eval.num_steps=45",
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
        "wandb.group=reasoning_method_smoke_5k",
        "wandb.job_type=train",
        f"wandb.tags=[smoke_5k,{method_name},{task},{mode}]",
    ]
    if experiment != "sudoku_sfm_truncated_adaptive":
        values += [
            f"algo.pure_continuous={str(mode == 'pure').lower()}",
            "+algo.input_embedding_dim=512",
            "+algo.output_embedding_dim=512",
        ]
    if mode == "hybrid":
        values.append(f"algo.hybrid_loss_weighting={hybrid_loss_weighting}")
    return {
        "method": method_name,
        "task": task,
        "mode": mode,
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
    parser.add_argument("--prepared", type=Path, default=ROOT / "data_cache/reasoning")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpus", nargs="+", default=["0", "2", "3", "4", "5", "6"])
    parser.add_argument("--project", default="reasoning-method-smoke-tests")
    parser.add_argument("--entity", default="rz-text-dlp")
    parser.add_argument(
        "--hybrid-loss-weighting",
        choices=("none", "likelihood"),
        default="none",
        help="plain CE (none) or continuous-time likelihood weighting",
    )
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
    prepared = args.prepared.resolve()

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
    profiles = OmegaConf.load(ROOT / "configs/sweeps/reasoning_tasks.yaml").profiles
    jobs = []
    # Interleave method families so configuration problems surface early.
    for task in ("zebra_4x4", "sudoku_hard", "sudoku_extreme", "maze_unique"):
        for method_name, spec in METHODS.items():
            if task in spec["tasks"]:
                jobs.append(
                    make_job(
                        matrix, profiles, root, prepared, method_name, task, spec,
                        args.seed, args.project, args.entity,
                        args.hybrid_loss_weighting,
                    )
                )
    plan = {
        "purpose": "cross_task_method_smoke_test",
        "max_steps": 5000,
        "evaluation_every_steps": 1000,
        "evaluation_examples": 128,
        "evaluation_nfe": 45,
        "seed": args.seed,
        "gpus": args.gpus,
        "project": args.project,
        "hybrid_loss_weighting": args.hybrid_loss_weighting,
        "jobs": jobs,
    }
    if len(jobs) != 18:
        raise AssertionError(f"expected 18 jobs, got {len(jobs)}")
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
    for task in ("zebra_4x4", "sudoku_extreme", "maze_unique"):
        if not (prepared / task / "manifest.json").exists():
            raise FileNotFoundError(prepared / task / "manifest.json")

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
