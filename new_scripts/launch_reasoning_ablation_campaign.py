#!/usr/bin/env python3
"""Durable shared-GPU queue for the restricted reasoning ablation matrix."""

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

from launch_reasoning_method_smoke import (
    snapshot,
    task_overrides as reasoning_task_overrides,
    write_json,
)
from sudoku_rank_ablation import ROOT


TASKS = (
    "sudoku_easy",
    "sudoku_medium",
    "sudoku_hard",
    "sudoku_extreme",
    "maze_unique",
)


def task_overrides(task, profile, root, prepared):
    if task in {"sudoku_easy", "sudoku_medium", "sudoku_hard"}:
        return [
            "data=sudoku",
            f"data.difficulty={task.removeprefix('sudoku_')}",
            f"data.cache_dir={root / 'data'}",
            "model.length=180",
            "loader.global_batch_size=256",
            "loader.eval_global_batch_size=128",
            "reasoning.batch_size=64",
            # Periodic task evaluation is handled by reasoning_eval. Disabling
            # Lightning's redundant full validation avoids the PyArrow worker
            # abort observed at step 5K in the earlier campaign.
            "trainer.limit_val_batches=0",
        ]
    return reasoning_task_overrides(task, profile, root, prepared)


def method_specs(matrix):
    """Expand the requested eight pure and eleven hybrid configurations."""
    specs = []
    for variant, entry in matrix.variants.items():
        for mode in entry.modes:
            experiment = entry.experiment
            overrides = list(entry.overrides)
            # The pure adaptive spherical baseline is S-FLM itself. Hybrid
            # spherical paths use the CANDI SLERP wrapper in the matrix.
            if variant == "slerp_truncated_adaptive" and mode == "pure":
                experiment = "sudoku_sfm_truncated_adaptive"
                overrides = []
            specs.append(
                dict(
                    variant=variant,
                    mode=mode,
                    label=str(entry.label),
                    experiment=experiment,
                    overrides=overrides,
                )
            )
    if len(specs) != 19:
        raise AssertionError(f"expected 19 method configurations, got {len(specs)}")
    return specs


def make_job(
    profiles,
    root,
    prepared,
    task,
    method,
    seed,
    project,
    entity,
    hybrid_loss_weighting,
):
    method_name = f"{method['variant']}_{method['mode']}"
    directory = root / "runs" / task / method_name / f"seed_{seed}"
    run_name = f"{task}_{method_name}_seed_{seed}"
    if method["mode"] == "hybrid":
        run_name += f"_loss_{hybrid_loss_weighting}"
    identity = {
        "campaign": root.name,
        "project": project,
        "task": task,
        "method": method,
        "seed": seed,
        "hybrid_loss_weighting": hybrid_loss_weighting,
    }
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:16]
    max_steps = 20000 if task.startswith("sudoku_") else profiles[task].max_steps
    values = [
        f"+experiment={method['experiment']}",
        *method["overrides"],
        "+callbacks=reasoning_eval",
        f"seed={seed}",
        *task_overrides(task, profiles.get(task), root, prepared),
        "trainer.devices=1",
        "trainer.num_nodes=1",
        f"trainer.max_steps={max_steps}",
        "loader.num_workers=4",
        "callbacks.reasoning_eval.every_n_train_steps=1000",
        "callbacks.reasoning_eval.subset_size=128",
        "callbacks.reasoning_eval.num_steps=45",
        "callbacks.checkpoint_every_n_steps.every_n_train_steps=5000",
        "callbacks.checkpoint_every_n_steps.save_top_k=0",
        "callbacks.checkpoint_every_n_steps.save_last=true",
        "callbacks.checkpoint_monitor.save_top_k=0",
        "callbacks.checkpoint_monitor.monitor=null",
        "checkpointing.resume_from_ckpt=true",
        f"checkpointing.save_dir={directory}",
        f"hydra.run.dir={directory}",
        f"wandb.project={project}",
        f"+wandb.entity={entity}",
        f"wandb.id={run_id}",
        f"wandb.name={run_name}",
        "wandb.group=reasoning_core_ablation",
        "wandb.job_type=train",
        f"wandb.tags=[core_ablation,{task},{method['variant']},{method['mode']}]",
    ]
    if method["experiment"] != "sudoku_sfm_truncated_adaptive":
        values += [
            f"algo.pure_continuous={str(method['mode'] == 'pure').lower()}",
            "+algo.input_embedding_dim=512",
            "+algo.output_embedding_dim=512",
        ]
    if method["mode"] == "hybrid":
        values.append(f"algo.hybrid_loss_weighting={hybrid_loss_weighting}")
    return {
        **identity,
        "method_name": method_name,
        "directory": str(directory),
        "run_id": run_id,
        "run_name": run_name,
        "wandb_url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        "overrides": values,
    }


def validate_overrides(jobs):
    """Compose every distinct job before any GPU process is started."""
    from hydra import compose, initialize_config_dir

    sys.path.insert(0, str(ROOT))
    import main as _main  # noqa: F401 -- registers project OmegaConf resolvers

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        for job in jobs:
            config = compose(config_name="config", overrides=job["overrides"])
            if config.trainer.devices != 1:
                raise ValueError("campaign jobs must each own exactly one GPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--prepared", type=Path, default=ROOT / "data_cache/reasoning"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=["sudoku_hard", "sudoku_extreme", "maze_unique"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Run only these manifest variants (all configured modes are retained).",
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["0", "2", "3", "4", "5", "6"]
    )
    parser.add_argument("--project", default="reasoning-candi-core-ablations")
    parser.add_argument("--entity", default="rz-text-dlp")
    parser.add_argument(
        "--hybrid-loss-weighting",
        choices=("none", "likelihood"),
        default="none",
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
                [sys.executable, *command],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(process.pid)
        return

    matrix = OmegaConf.load(ROOT / "configs/sweeps/sudoku_rank.yaml")
    profiles = OmegaConf.load(
        ROOT / "configs/sweeps/reasoning_tasks.yaml"
    ).profiles
    methods = method_specs(matrix)
    if args.variants:
        unknown = sorted(set(args.variants) - set(matrix.variants))
        if unknown:
            parser.error(f"unknown variants: {', '.join(unknown)}")
        selected = set(args.variants)
        methods = [method for method in methods if method["variant"] in selected]
    jobs = [
        make_job(
            profiles,
            root,
            prepared,
            task,
            method,
            seed,
            args.project,
            args.entity,
            args.hybrid_loss_weighting,
        )
        for seed in args.seeds
        for task in args.tasks
        for method in methods
    ]
    plan = {
        "purpose": "restricted_reasoning_core_ablation",
        "tasks": args.tasks,
        "methods_per_task": len(methods),
        "seeds": args.seeds,
        "training_runs": len(jobs),
        "gpus": args.gpus,
        "queue_policy": "shared_dynamic_fifo",
        "project": args.project,
        "hybrid_loss_weighting": args.hybrid_loss_weighting,
        "monitoring": {"interval": 1000, "examples": 128, "nfe": 45},
        "jobs": jobs,
    }
    validate_overrides(jobs)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return

    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign_path = root / "campaign.json"
    if campaign_path.exists() and json.loads(campaign_path.read_text()) != plan:
        raise ValueError("existing campaign differs; use a new --root")
    write_json(campaign_path, plan)
    source = snapshot(root)

    for task in args.tasks:
        if task in {"sudoku_easy", "sudoku_medium", "sudoku_hard"}:
            continue
        manifest = prepared / task / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
    cache_root = root / "data"
    cache_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if task not in {"sudoku_easy", "sudoku_medium", "sudoku_hard"}:
            continue
        difficulty = task.removeprefix("sudoku_")
        name = f"sudoku_{difficulty}_train48000_valid2000_seed42"
        destination = cache_root / name
        candidates = [
            args.hard_cache if difficulty == "hard" else ROOT / "data_cache" / name,
            Path("/home/patrick/.cache/duo/data_cache") / name,
        ]
        source_cache = next((path for path in candidates if path.exists()), None)
        if not destination.exists() and source_cache is not None:
            destination.symlink_to(source_cache.resolve())
        if not destination.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "new_scripts/prepare_sudoku.py"),
                    "--difficulty",
                    difficulty,
                    "--cache-dir",
                    str(cache_root),
                ],
                cwd=ROOT,
                check=True,
            )

    states = {
        str(index): (
            "complete"
            if (Path(job["directory"]) / "train.complete").exists()
            else "queued"
        )
        for index, job in enumerate(jobs)
    }
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

    def gpu_idle(gpu):
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
        return memory < 1024 and utilization < 10

    def worker(gpu):
        while True:
            while not gpu_idle(gpu):
                time.sleep(30)
            with mutex:
                queued = [i for i in range(len(jobs)) if states[str(i)] == "queued"]
                if not queued:
                    return
                index = queued[0]
                states[str(index)] = f"starting on GPU {gpu}"
                save()
            job = jobs[index]
            directory = Path(job["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            try:
                if shutil.disk_usage(root).free < 8 * 2**30:
                    raise RuntimeError("less than 8 GiB free disk")
                command = [sys.executable, str(source / "main.py"), *job["overrides"]]
                write_json(
                    directory / "run.json",
                    {**job, "gpu": gpu, "command": command},
                )
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
                    {
                        "returncode": returncode,
                        "seconds": time.monotonic() - started,
                    },
                )
                if returncode:
                    raise RuntimeError(f"training exit {returncode}")
                (directory / "train.complete").touch()
                status = "complete"
            except Exception as error:
                status = f"failed: {error}"
            with mutex:
                states[str(index)] = status
                save()

    save()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    print(json.dumps(states, indent=2), flush=True)


if __name__ == "__main__":
    main()
