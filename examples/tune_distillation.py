"""Build per-sampler Optuna cases for completed fixed-path distillation runs."""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from core.sweeps.nested_tuning import validate


def policies(gaussian):
    temperature = {
        "algorithm.predictor.temperature": {"type": "float", "low": 0.5, "high": 1.5}
    }
    native = dict(
        _target_="core.samplers.hybrid.HybridSampler",
        steps=3,
        final_denoise="greedy",
        greedy_clean_estimate=False,
    )
    out = [
        dict(
            name="native",
            sampler=native,
            space=temperature,
            initial_points=[{"algorithm.predictor.temperature": 1.0}],
            trials=16,
        ),
        dict(
            name="native_greedy",
            sampler={**native, "greedy_clean_estimate": True},
            space={"sampler.final_denoise": ["greedy", "sample"]},
            initial_points=[{"sampler.final_denoise": x} for x in ("greedy", "sample")],
            trials=2,
        ),
        dict(
            name="eb_greedy",
            sampler=dict(
                _target_="core.samplers.parallel_entropy_bounded.ParallelEntropyBoundedSampler",
                steps=3,
                freeze=True,
                greedy_clean_estimate=True,
                entropy_bound=0.1,
            ),
            space={
                "sampler.entropy_bound": {
                    "type": "float",
                    "low": 0.02,
                    "high": 0.5,
                    "log": True,
                },
                **temperature,
            },
            initial_points=[
                {"sampler.entropy_bound": 0.1, "algorithm.predictor.temperature": 1.0}
            ],
            trials=16,
        ),
    ]
    if gaussian:
        out.append(
            dict(
                name="forward_corruption",
                sampler=dict(
                    _target_="core.samplers.composed.ComposedSampler",
                    steps=3,
                    tokens=dict(
                        _target_="core.samplers.policies.TokenSelection", mode="sample"
                    ),
                    transition=dict(
                        _target_="core.samplers.policies.ForwardCorruption"
                    ),
                    final_denoise="greedy",
                ),
                space={**temperature, "sampler.tokens.mode": ["sample", "greedy"]},
                initial_points=[
                    {"algorithm.predictor.temperature": 1.0, "sampler.tokens.mode": x}
                    for x in ("sample", "greedy")
                ],
                trials=16,
            )
        )
    return [dict(**p, budget_parameter="steps") for p in out]


def build(root, *, prefix="fewstep-3000-", expected=27, exclude=()):
    cases = []
    partitions = {}
    seen = set()
    for directory in sorted(root.glob("run_job*")):
        params = directory / "params.json"
        result = directory / "result.json"
        if (
            not params.exists()
            or not result.exists()
            or not result.stat().st_size
            or (directory / "error.txt").exists()
        ):
            continue
        try:
            job = json.loads(params.read_text())
        except json.JSONDecodeError:
            if expected is None:  # Ray may be rewriting metadata while polling.
                continue
            raise
        if (
            job["name"] in exclude
            or job["stage"] != "train"
            or not job["name"].startswith(prefix)
            or "-inherited-" in job["name"]
        ):
            continue
        try:
            last = json.loads(result.read_text().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            if expected is None:
                continue
            raise
        saved = directory / last["checkpoint_dir_name"]
        checkpoint = saved / "checkpoints/last.ckpt"
        config_path = saved / "config.yaml"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config = OmegaConf.load(config_path)
        task = next(
            t
            for t in ("sudoku_hard", "text8", "tinystories")
            if f"-{t}-" in job["name"]
        )
        overrides = {}
        if "num_eval_samples" in config.task:
            overrides["task.num_eval_samples"] = 100000000
        if task == "text8":
            overrides["task.min_word_lengths"] = [5, 6]
        if task == "tinystories":
            overrides["task.calculate_bleu"] = True
        if task not in partitions:
            excluded = set()
            if "num_eval_samples" in config.task:
                for n in (16, 128, 256):
                    previous = copy.deepcopy(config.task)
                    previous.num_eval_samples = n
                    excluded.update(
                        instantiate(previous).evaluation_dataset("validation").indices
                    )
                config.task.num_eval_samples = 100000000
            else:
                excluded.update(range(256))
            full = instantiate(config.task).evaluation_dataset("validation")
            # Full TinyStories/text8 subsets are sorted, so positions equal raw IDs.
            available = [i for i in range(len(full)) if i not in excluded]
            indices = np.random.default_rng(7351).permutation(available).tolist()
            if len(indices) < 384:
                raise ValueError(f"Insufficient fresh validation examples: {task}")
            partitions[task] = dict(
                tuning_indices=indices[:128], test_indices=indices[128:384]
            )
        metric = {
            "sudoku_hard": "accuracy",
            "text8": "generated/by_min_length/5/aggregate/mean_per_sample/unique_valid",
            "tinystories": "corpus_bleu",
        }[task]
        gaussian = job["name"].startswith(
            ("fewstep-3000-gaussian-", "fewstep-3000-dmd-")
        )
        for nfe in (4, 16):
            for policy in policies(gaussian):
                case = dict(
                    name=f"{job['name']}-nfe-{nfe}-{policy['name']}",
                    config=str(config_path),
                    training=dict(checkpoint=str(checkpoint), space={}, trials=1),
                    evaluation=dict(
                        overrides=overrides,
                        nfe=nfe,
                        seeds=[41, 42],
                        test_seeds=[1041, 1042],
                        direction="maximize",
                        score=dict(kind="metric", metric=metric),
                        **partitions[task],
                    ),
                    inference=[policy],
                    wandb=dict(
                        entity="rz-text-dlp",
                        project="distillation-smoke-20261026",
                        group="fewstep-3000-sampler-tuning",
                    ),
                    resources=dict(cpu=4, gpu=1),
                )
                validate(case)
                cases.append(case)
        seen.add(job["name"])
    if expected is not None and len(seen) != expected:
        raise ValueError(
            f"Expected {expected} completed checkpoints; found {len(seen)}"
        )
    return cases


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", default="fewstep-3000-")
    p.add_argument("--expected", type=int, default=27)
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument(
        "--wait",
        action="store_true",
        help="Wait for requested checkpoints before building",
    )
    p.add_argument(
        "--run-address", help="Run the existing nested tuner on this Ray address"
    )
    args = p.parse_args()
    if args.expected < 1:
        p.error("expected must be positive")
    if args.wait:
        import time

        while True:
            cases = build(
                args.runs, prefix=args.prefix, expected=None, exclude=args.exclude
            )
            ready = len({case["training"]["checkpoint"] for case in cases})
            if ready == args.expected:
                break
            if ready > args.expected:
                raise ValueError("More checkpoints matched than expected")
            print(f"Waiting for checkpoints: {ready}/{args.expected}", flush=True)
            time.sleep(30)
    else:
        cases = build(
            args.runs, prefix=args.prefix, expected=args.expected, exclude=args.exclude
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"{len(cases)} independent checkpoint/NFE/sampler cases")

    if args.run_address:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sweeps.nested_tuning",
                str(args.output),
                "--output",
                str(args.output.parent),
                "--address",
                args.run_address,
                "--case-concurrency",
                "4",
                "--concurrency",
                "1",
            ],
            check=True,
        )
