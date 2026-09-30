"""Materialize and execute the SDM Sudoku matrix; every experiment uses W&B.

Plan is the default. Training and evaluation are explicit phases, never inferred
from the existence of partial checkpoints. Results require all five seeds.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import queue
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
CONCENTRATIONS = [
    ("cl_02_05_02", "constant_linear", 0.2, 0.5, 0.2),
    ("cl_02_075_02", "constant_linear", 0.2, 0.75, 0.2),
    ("cl_04_075_08", "constant_linear", 0.4, 0.75, 0.8),
    ("c_05", "constant", 0.5, 0.5, 0.8),
    ("c_025", "constant", 0.25, 0.25, 0.8),
]


def experiment_matrix(seeds, full=False):
    for seed, sc, adaptive, concentration in itertools.product(
        seeds,
        (False, True),
        (False, True),
        CONCENTRATIONS if full else CONCENTRATIONS[2:3],
    ):
        label, kind, nu0, nu1, pivot = concentration
        yield {
            "name": f"simplex_{label}_sc{int(sc)}_adaptive{int(adaptive)}_seed{seed}",
            "simplex": True,
            "adaptive": adaptive,
            "overrides": [
                "+experiment=sudoku_simplex",
                f"seed={seed}",
                f"self_conditioning.enabled={str(sc).lower()}",
                f"time_sampling.adaptive={str(adaptive).lower()}",
                f"trainer.max_steps={38462 if sc else 50000}",
                f"algo.concentration={kind}",
                f"algo.nu0={nu0}",
                f"algo.nu1={nu1}",
                f"algo.pivot={pivot}",
            ],
        }
    for seed, sc, adaptive, method in itertools.product(
        seeds,
        (False, True),
        (False, True),
        ("mdlm_ce", "mdlm_elbo", "uniform"),
    ):
        experiment = (
            "sudoku_simplex_uniform" if method == "uniform" else "sudoku_simplex_mdlm"
        )
        overrides = [
            f"+experiment={experiment}",
            f"seed={seed}",
            f"self_conditioning.enabled={str(sc).lower()}",
            f"time_sampling.adaptive={str(adaptive).lower()}",
            f"trainer.max_steps={38462 if sc else 50000}",
        ]
        if method == "mdlm_ce":
            overrides.append("algo.loss_type=ce")
        yield {
            "name": f"{method}_sc{int(sc)}_adaptive{int(adaptive)}_seed{seed}",
            "simplex": False,
            "adaptive": adaptive,
            "overrides": overrides,
        }


def commands(entry, args):
    directory = args.output / entry["name"]
    weights = getattr(args, "weights", "ema")
    run_name = entry["name"] + ("_current" if weights == "current" else "")
    common = [
        sys.executable,
        str(ROOT / "main.py"),
        *entry["overrides"],
        f"wandb.project={args.project}",
        "wandb.mode=online",
        "require_wandb=true",
        f"eval.disable_ema={str(weights == 'current').lower()}",
        f"wandb.tags=[simplex-paper,weights-{weights}]",
        f"data.cache_dir={args.output / 'data_cache'}",
        "trainer.devices=1",
        "strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy",
        "+strategy.device=cuda:0",
        "~strategy.find_unused_parameters",
        f"hydra.run.dir={directory}",
        f"checkpointing.save_dir={directory}",
        f"wandb.name={run_name}",
    ]
    if args.entity:
        common.append(f"+wandb.entity={args.entity}")
    train = common + ["mode=train"]
    evaluations = []
    for grid, variant in itertools.product(
        ("linear", "cosine", "adaptive") if entry["adaptive"] else ("linear", "cosine"),
        (0.0, 0.2, 1.0) if entry["simplex"] else (False, True),
    ):
        label = f"{grid}_{'churn' if entry['simplex'] else 'pc'}{variant}"
        destination = directory / "evaluation" / label
        setting = (
            f"sampling.churn={variant}"
            if entry["simplex"]
            else f"sampling.predict_correct={str(variant).lower()}"
        )
        evaluations.append(
            common
            + [
                "mode=sudoku_eval",
                f"sampling.time_grid={grid}",
                setting,
                f"eval.checkpoint_path={directory / 'checkpoints' / 'final.ckpt'}",
                f"sudoku.output_dir={destination}",
                f"hydra.run.dir={destination}",
                f"wandb.name={run_name}_{label}",
                "wandb.job_type=evaluation",
            ]
        )
    return {"name": entry["name"], "train": train, "evaluations": evaluations}


def summarize(args):
    groups = {}
    for path in args.output.glob("*/evaluation/*/results.json"):
        run = path.parents[2].name
        family, seed = run.rsplit("_seed", 1)
        result = json.loads(path.read_text())
        if result["num_total"] != 5000:
            raise ValueError(f"{path}: expected full 5000-puzzle evaluation")
        groups.setdefault((family, path.parent.name), {})[int(seed)] = (
            result["accuracy"] * 100
        )
    summaries = []
    for (family, sampling), values in groups.items():
        complete = set(values) == set(args.seeds) and len(args.seeds) == 5
        summaries.append(
            {
                "method": family,
                "sampling": sampling,
                "seeds": values,
                "complete_five_seed_evaluation": complete,
                "paper_accuracy_comparison": "pending",
                "mean_accuracy_percent": statistics.mean(values.values()),
                "std_accuracy_percent": statistics.pstdev(values.values())
                if len(values) > 1
                else None,
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


def run_queue(matrix, args):
    """Each GPU owns one subprocess at a time; failures remain explicit."""
    pending = queue.Queue()
    for entry in matrix:
        pending.put(entry)

    def worker(gpu):
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4"
        )
        while True:
            try:
                entry = pending.get_nowait()
            except queue.Empty:
                return
            directory = args.output / entry["name"]
            directory.mkdir(parents=True, exist_ok=True)
            status = {"gpu": gpu, "name": entry["name"], "state": "training"}
            status_path = directory / "queue_status.json"
            status_path.write_text(json.dumps(status, indent=2))
            try:
                for index, command in enumerate(
                    [entry["train"], *entry["evaluations"]]
                ):
                    if shutil.disk_usage(args.output).free < 20 * 1024**3:
                        status.update(
                            state="blocked", reason="Less than 20 GiB free storage"
                        )
                        status_path.write_text(json.dumps(status, indent=2))
                        raise RuntimeError(
                            "Reproduction paused before exhausting shared storage"
                        )
                    status["state"] = "training" if index == 0 else "evaluating"
                    status["evaluation_index"] = index
                    status_path.write_text(json.dumps(status, indent=2))
                    with (directory / f"process_{index:02d}.log").open("w") as log:
                        subprocess.run(
                            command,
                            cwd=ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                status["state"] = "complete"
            except subprocess.CalledProcessError as exc:
                status.update(state="failed", exit_code=exc.returncode)
                status_path.write_text(json.dumps(status, indent=2))
                raise
            status_path.write_text(json.dumps(status, indent=2))

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    summarize(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("plan", "train", "evaluate", "summarize", "run"),
        default="plan",
    )
    parser.add_argument(
        "--full", action="store_true", help="All five concentration schedules"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "simplex-paper")
    parser.add_argument("--project", default="simplex-sudoku")
    parser.add_argument("--entity")
    parser.add_argument(
        "--weights",
        choices=("ema", "current"),
        default="ema",
        help="Weights used by training-time sampling, validation, and final evaluation",
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[1], help="GPU workers for phase=run"
    )
    parser.add_argument(
        "--match", default="", help="Restrict to run names containing this string"
    )
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus must list distinct nonnegative device indices")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "summarize":
        summarize(args)
        return
    matrix = [
        commands(entry, args)
        for entry in experiment_matrix(args.seeds, args.full)
        if args.match in entry["name"]
    ]
    priority = [
        "simplex_cl_04_075_08_sc1_adaptive0_seed1",
        "simplex_cl_04_075_08_sc0_adaptive0_seed1",
        "mdlm_ce_sc1_adaptive0_seed1",
        "uniform_sc1_adaptive0_seed1",
    ]
    matrix.sort(
        key=lambda entry: (
            priority.index(entry["name"])
            if entry["name"] in priority
            else len(priority)
        )
    )
    (args.output / "manifest.json").write_text(json.dumps(matrix, indent=2))
    print(
        f"{len(matrix)} training runs; {sum(len(e['evaluations']) for e in matrix)} evaluation runs"
    )
    if args.phase == "plan":
        return
    if args.phase == "run":
        run_queue(matrix, args)
        return
    for entry in matrix:
        for command in (
            [entry["train"]] if args.phase == "train" else entry["evaluations"]
        ):
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
