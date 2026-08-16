#!/usr/bin/env python3
"""Evaluate AR checkpoints on one paired 128-example GSM8K temperature sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import algos  # noqa: F401: populate the algorithm registry
import dataloader
from algos import algo_registry
from tasks.evaluation import generate_gsm8k_records, score_gsm8k_records
from tasks.gsm8k import get_gsm8k_dataset


def parse_checkpoint(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from error
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return label, path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.2]
    )
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(temperature < 0 for temperature in args.temperatures):
        raise ValueError("temperatures must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                "+experiment=tinygsm_ar_20m",
                "trainer.devices=1",
                f"data.cache_dir={REPO_ROOT / 'data_cache'}",
                "sampling.predictor=ar_cached",
                "sampling.steps=32",
            ],
        )
    tokenizer = dataloader.get_tokenizer(config)
    dataset = get_gsm8k_dataset(config, tokenizer)
    if args.subset_size > len(dataset):
        raise ValueError(f"subset size {args.subset_size} exceeds dataset size {len(dataset)}")
    rng = np.random.default_rng(args.subset_seed)
    indices = sorted(rng.choice(len(dataset), args.subset_size, replace=False).tolist())
    (args.output_dir / "subset_indices.json").write_text(json.dumps(indices, indent=2))

    algo_cls = algo_registry.get(config.algo.name)
    summary = []
    for label, checkpoint in args.checkpoint:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_step = int(checkpoint_data.get("global_step", -1))
        del checkpoint_data
        model = algo_cls.load_from_checkpoint(checkpoint, tokenizer=tokenizer, config=config)
        model.to("cuda")
        model._eval_mode()
        for temperature in args.temperatures:
            with open_dict(config):
                config.algo.temp = float(temperature)
            torch.manual_seed(args.sampling_seed)
            torch.cuda.manual_seed_all(args.sampling_seed)
            records = generate_gsm8k_records(
                model=model,
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_steps=config.sampling.steps,
                indices=indices,
            )
            results = score_gsm8k_records(
                records,
                method="code",
                timeout=float(config.gsm8k.timeout),
                bootstrap_size=1000,
                seed=args.subset_seed,
            )
            results.update(
                checkpoint_label=label,
                checkpoint_path=str(checkpoint),
                checkpoint_step=checkpoint_step,
                temperature=float(temperature),
                subset_seed=args.subset_seed,
                sampling_seed=args.sampling_seed,
            )
            destination = args.output_dir / f"{label}_temp_{temperature:g}.json"
            destination.write_text(json.dumps(results, indent=2))
            row = {key: value for key, value in results.items() if key != "records"}
            summary.append(row)
            print(
                f"{label} step={checkpoint_step} temp={temperature:g}: "
                f"{results['num_correct']}/{results['num_total']} "
                f"({100 * results['accuracy']:.2f}%)",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
