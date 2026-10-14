"""Build inference tuning manifests from completed three-task sweep artifacts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build(results_csv, *, all_methods=False):
    methods = {
        f"{noise}-sc-{sc}_20m"
        for noise in ("masked", "uniform")
        for sc in ("off", "on")
    }
    jobs = []
    seen = set()
    for row in csv.DictReader(Path(results_csv).open()):
        insertion = row["policy"] == "insertion_128_sample"
        if row["policy"] not in {"native_128", "insertion_128_sample"}:
            continue
        if not all_methods and row["method"] not in methods:
            continue
        identity = (row["task"], row["method"])
        if identity in seen:
            raise ValueError(f"Duplicate checkpoint: {identity}")
        seen.add(identity)
        config_path = Path(row["source"]).parent / "config.yaml"
        config = OmegaConf.load(config_path)
        task = instantiate(config.task)
        previous = task.evaluation_dataset("validation")
        excluded = (
            set(previous.indices) if hasattr(previous, "indices") else set(range(128))
        )
        overrides = {}
        if "num_eval_samples" in config.task:
            overrides["task.num_eval_samples"] = 100000000
            config.task.num_eval_samples = 100000000
        if row["task"] == "text8":
            overrides["task.min_word_lengths"] = [5, 6]
        dataset = instantiate(config.task).evaluation_dataset("validation")
        available = [i for i in range(len(dataset)) if i not in excluded]
        ids = np.random.default_rng(7351).permutation(available).tolist()
        if len(ids) < 384:
            raise ValueError(f"Insufficient held-out examples for {identity}")
        score = {"kind": "metric", "metric": "accuracy"}
        if row["task"] == "text8":
            score["metric"] = (
                "generated/by_min_length/5/aggregate/mean_per_sample/unique_valid"
            )
        elif row["task"] == "tinystories":
            overrides["task.calculate_bleu"] = True
            score = {"kind": "metric", "metric": "corpus_bleu"}
        policies = (
            ("insertion_sample", "insertion_greedy")
            if insertion
            else ("native", "native_greedy", "eb_greedy")
        )
        job = {
            "name": "-".join(identity),
            "config": str(config_path),
            "training": {
                "checkpoint": config.evaluation.checkpoint,
                "space": {},
                "trials": 1,
            },
            "evaluation": {
                "overrides": overrides,
                "nfe": 128,
                "seeds": [41, 42],
                "test_seeds": [1041, 1042],
                "direction": "maximize",
                "score": score,
                "tuning_indices": ids[:128],
                "test_indices": ids[128:384],
            },
            "inference": [],
            "wandb": {
                "entity": "rz-text-dlp",
                "project": "inference-tuning-three-tasks",
            },
        }
        jobs.append(job)
        for policy in policies:
            greedy = policy != "native"
            sampler = {
                "_target_": "core.samplers.hybrid.HybridSampler",
                "steps": 127,
                "epsilon": 1e-5,
                "final_denoise": "greedy",
                "greedy_clean_estimate": greedy,
            }
            space = {
                "algorithm.predictor.temperature": {
                    "type": "float",
                    "low": 0.5,
                    "high": 1.5,
                }
            }
            initial = [{"algorithm.predictor.temperature": 1.0}]
            if greedy and not insertion:
                space = {"sampler.final_denoise": ["greedy", "sample"]}
                initial = [
                    {"sampler.final_denoise": value} for value in ("greedy", "sample")
                ]
            if policy == "eb_greedy":
                sampler = {
                    "_target_": "core.samplers.parallel_entropy_bounded.ParallelEntropyBoundedSampler",
                    "steps": 127,
                    "entropy_bound": 0.1,
                    "freeze": True,
                    "greedy_clean_estimate": True,
                }
                space = {
                    "sampler.entropy_bound": {
                        "type": "float",
                        "low": 0.02,
                        "high": 0.5,
                        "log": True,
                    }
                }
                initial = [{"sampler.entropy_bound": 0.1}]
            if insertion:
                sampler = OmegaConf.to_container(config.sampler, resolve=True)
                sampler["final_denoise"] = (
                    "sample" if policy == "insertion_sample" else "greedy"
                )
            job["inference"].append(
                {
                    "name": policy,
                    "sampler": sampler,
                    "space": space,
                    "initial_points": initial,
                    "budget_parameter": "steps",
                    "trials": 2 if policy == "native_greedy" else 16,
                }
            )
    expected = 60 if all_methods else 12
    if len(seen) != expected:
        raise ValueError(f"Expected {expected} checkpoints, found {len(seen)}")
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(build(args.results, all_methods=args.all_methods), indent=2)
    )
