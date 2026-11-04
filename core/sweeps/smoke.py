"""Isolated train/checkpoint/sampler checks using the ordinary Ray workers.

This does not perform tuning or evaluate final-test examples. Each policy uses
its first initial point (or a deterministic value from its search space).
"""

import argparse
import copy
import json
import math
from pathlib import Path
import traceback

from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.cluster.slurm import positive
from core.sweeps.budget import sampler_steps
from core.sweeps.nested_tuning import execute, save, validate


def parameters(policy):
    point = copy.deepcopy(next(iter(policy.get("initial_points", [])), {}))
    for key, space in policy["space"].items():
        if key not in point:
            point[key] = space[0] if isinstance(space, list) else space["low"]
    return point


def smoke_config(path, train_steps, examples, no_compile):
    cfg = OmegaConf.load(path)
    overrides = {
        "trainer.max_steps": train_steps,
        "trainer.accumulate_grad_batches": 1,
        "trainer.devices": 1,
        "trainer.num_nodes": 1,
        "trainer.strategy": "auto",
        "trainer.num_sanity_val_steps": 0,
        "trainer.limit_val_batches": 0,
        "trainer.log_every_n_steps": 1,
        "loader.global_batch_size": examples,
        "loader.eval_batch_size": examples,
        "loader.num_workers": 0,
        "checkpoint.resume": None,
        "checkpoint.every_n_steps": 1,
        "checkpoint.save_top_k": 1,
        "callback": None,
        "wandb": None,
    }
    if no_compile:
        overrides.update(
            {"compile": False, "training.performance.compile.enabled": False}
        )
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    # Preserve data, tokenizer, architecture and objective. No cache regeneration.
    return cfg


def policy_budget(cfg, policy, point, steps):
    policy = copy.deepcopy(policy)
    trial = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    trial.sampler = policy["sampler"]
    for key, value in point.items():
        OmegaConf.update(trial, key, value, force_add=True)
    if policy["budget_parameter"] is None:
        if (
            trial.sampler["_target_"]
            != "core.samplers.autoregressive.AutoregressiveSampler"
        ):
            raise ValueError("No one-step smoke adapter for this non-iterative sampler")
        policy["sampler"]["max_new_tokens"] = steps
        return policy, steps
    sampler = instantiate(trial.sampler)
    # Find the minimum budget that produces exactly the requested transitions.
    # This includes corrector/bootstrap calls, terminal readout and all blocks.
    blocks = 1
    if OmegaConf.select(trial, "block_diffusion.enabled", default=False):
        blocks = math.ceil(trial.task.length / trial.block_diffusion.block_size)
    for per_block in range(2, 4097):
        nfe = per_block * blocks
        try:
            actual = sampler_steps(trial, sampler, nfe)
        except ValueError:
            continue
        if actual == steps:
            return policy, nfe
        if actual > steps:
            break
    raise ValueError(f"Cannot allocate exactly {steps} steps to {policy['name']}")


def run(manifest, output, train_steps=1, steps=1, examples=1, no_compile=False):
    cases = json.loads(Path(manifest).read_text())
    if not cases or len({s["name"] for s in cases}) != len(cases):
        raise ValueError("Manifest must have nonempty, unique cases")
    for spec in cases:
        if Path(spec["name"]).name != spec["name"] or spec["name"] in {".", ".."}:
            raise ValueError("Unsafe case name")
        validate(spec)
        for policy in spec["inference"]:
            if Path(policy["name"]).name != policy["name"] or policy["name"] in {
                ".",
                "..",
            }:
                raise ValueError("Unsafe policy name")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "smoke_only_not_benchmark",
        "train_steps": train_steps,
        "sampler_steps": steps,
        "examples": examples,
        "cases": [],
    }
    for original in cases:
        spec = copy.deepcopy(original)
        destination = output / spec["name"]
        destination.mkdir()
        row = {"name": spec["name"], "training": "pending", "policies": []}
        report["cases"].append(row)
        save(destination / "original-case.json", original)
        try:
            cfg = smoke_config(spec["config"], train_steps, examples, no_compile)
            evaluation = copy.deepcopy(spec["evaluation"])
            # Smoke never sees final-test IDs/seeds.
            evaluation.pop("test_indices", None)
            evaluation.pop("test_seeds", None)
            evaluation.pop("test_dataset", None)
            evaluation["indices"] = evaluation.pop("tuning_indices")[:examples]
            evaluation["seeds"] = evaluation["seeds"][:1]
            resources = spec.get("resources", {"cpu": 4, "gpu": 1})
            # Evaluation overrides must not undo the small smoke batch.
            evaluation.setdefault("overrides", {})["loader.eval_batch_size"] = examples
            OmegaConf.save(cfg, destination / "smoke-config.yaml")
            checkpoint = execute(
                "train",
                {
                    "config": OmegaConf.to_container(cfg, resolve=False),
                    "parameters": {},
                    "validation": evaluation,
                    "resources": resources,
                },
                destination / "training",
            )
            row["training"] = "passed"
        except Exception:
            row.update(training="failed", error=traceback.format_exc())
            save(output / "smoke-report.json", report)
            continue
        for policy in spec["inference"]:
            result = {"name": policy["name"]}
            row["policies"].append(result)
            try:
                point = parameters(policy)
                selected, nfe = policy_budget(cfg, policy, point, steps)
                result.update(parameters=point, nfe_budget=nfe)
                metrics = execute(
                    "evaluate",
                    {
                        "checkpoint": checkpoint,
                        "policy": selected,
                        "parameters": point,
                        "evaluation": {**evaluation, "nfe": nfe},
                        "resources": resources,
                    },
                    destination / "evaluation" / policy["name"],
                )
                if not math.isfinite(metrics["score"]):
                    raise ValueError("Nonfinite smoke evaluation score")
                result.update(status="passed", **metrics)
            except Exception:
                result.update(status="failed", error=traceback.format_exc())
            save(output / "smoke-report.json", report)
    report["passed"] = all(
        row["training"] == "passed"
        and len(row["policies"]) == len(spec["inference"])
        and all(p["status"] == "passed" for p in row["policies"])
        for row, spec in zip(report["cases"], cases)
    )
    save(output / "smoke-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--train-steps", type=positive, default=1)
    parser.add_argument("--sampler-steps", type=positive, default=1)
    parser.add_argument("--examples", type=positive, default=1)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    import ray

    ray.init(address=args.address)
    try:
        result = run(
            args.manifest,
            args.output,
            args.train_steps,
            args.sampler_steps,
            args.examples,
            args.no_compile,
        )
    finally:
        ray.shutdown()
    if not result["passed"]:
        raise SystemExit("Smoke failed; see smoke-report.json")


if __name__ == "__main__":
    main()
