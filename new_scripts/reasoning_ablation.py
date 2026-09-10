#!/usr/bin/env python3
"""Run the existing 16-model matrix on prepared reasoning task profiles."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys

from omegaconf import OmegaConf
from sudoku_rank_ablation import ROOT, call_main, overrides as sudoku_overrides

sys.path.insert(0, str(ROOT))
from tasks.reasoning_data import prepare, file_hash


def jobs(matrix, profiles, args):
    for profile, variant, seed in itertools.product(
        args.profiles or profiles,
        args.variants or matrix.variants,
        args.seeds or ([1] if args.command == "smoke" else matrix.seeds),
    ):
        for mode in args.modes or matrix.modes:
            if mode in matrix.variants[variant].modes:
                yield dict(profile=profile, variant=variant, mode=mode, seed=seed)


def overrides(matrix, profiles, job, root, prepared, smoke=False):
    profile = profiles[job["profile"]]
    _, values = sudoku_overrides(matrix, dict(job, difficulty="easy"), root, smoke)
    values = [
        v
        for v in values
        if not v.startswith(
            (
                "data.difficulty=",
                "data.num_train=",
                "data.num_valid=",
                "checkpointing.save_dir=",
                "hydra.run.dir=",
            )
        )
    ]
    directory = (
        root / job["profile"] / job["variant"] / job["mode"] / f"seed_{job['seed']}"
    )
    batch = 2 if smoke else profile.batch_size
    values += [
        f"data={profile.task}",
        f"data.prepared_dir={prepared / job['profile']}",
        f"model.length={profile.length}",
        f"data.prompt_length={profile.prompt_length}",
        f"checkpointing.save_dir={directory}",
        f"hydra.run.dir={directory}",
        f"loader.global_batch_size={batch}",
        f"loader.eval_global_batch_size={batch}",
        f"reasoning.batch_size={batch}",
        f"trainer.max_steps={2 if smoke else profile.max_steps}",
    ]
    if smoke:
        values += [
            "data.limit_train=8",
            "data.limit_validation=4",
            "reasoning.max_examples=4",
            "model.hidden_size=16",
            "model.cond_dim=16",
            "model.n_heads=1",
        ]
    return directory, values


def aggregate(root):
    rows = []
    for path in sorted(root.glob("*/*/*/seed_*/run.json")):
        meta = json.loads(path.read_text())
        for result_path in sorted(path.parent.glob("eval/*/nfe_*/results.json")):
            result = json.loads(result_path.read_text())
            rows.append(
                {
                    **meta["job"],
                    "smoke": meta["smoke"],
                    "evaluation_profile": result_path.parent.parent.name,
                    "requested_nfe": int(result_path.parent.name[4:]),
                    **{
                        k: v
                        for k, v in result.items()
                        if k not in ("records", "by_layout")
                    },
                }
            )
    if not rows:
        raise ValueError(f"no results under {root}")

    def write(name, records):
        with (root / name).open("w") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(dict.fromkeys(k for row in records for k in row)),
            )
            writer.writeheader()
            writer.writerows(records)

    write("summary.csv", rows)
    keys = [
        "profile",
        "variant",
        "mode",
        "evaluation_profile",
        "requested_nfe",
        "smoke",
    ]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for key, group in groups.items():
        row = dict(zip(keys, key), num_seeds=len(group))
        for metric in group[0]:
            if metric in keys or metric in ("seed", "num_total"):
                continue
            numbers = [
                r[metric] for r in group if isinstance(r.get(metric), (int, float))
            ]
            if numbers:
                row[metric + "_mean"] = statistics.mean(numbers)
                row[metric + "_std"] = (
                    statistics.stdev(numbers) if len(numbers) > 1 else None
                )
        summaries.append(row)
    write("summary_by_variant.csv", summaries)
    print(root / "summary.csv")


def run(matrix, profiles, job, args, gpu):
    smoke = args.command == "smoke"
    directory, values = overrides(
        matrix, profiles, job, args.root, args.prepared, smoke
    )
    source = json.loads((args.prepared / job["profile"] / "manifest.json").read_text())
    if not smoke and source["source_limit"] is not None:
        raise ValueError(
            "full sweeps require complete prepared releases; this is a smoke subset"
        )
    if source["full_training"]:
        values += [
            "data.full_training=true",
            "trainer.limit_val_batches=0",
            "callbacks.checkpoint_monitor.save_top_k=0",
            "callbacks.checkpoint_monitor.monitor=null",
        ]
    metadata = dict(
        job=job,
        overrides=values,
        smoke=smoke,
        dataset_fingerprint=source["fingerprint"],
    )
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    run_path = directory / "run.json"
    if (
        run_path.exists()
        and json.loads(run_path.read_text())["fingerprint"] != fingerprint
    ):
        raise ValueError(f"existing run has different data/config: {directory}")
    run_path.write_text(json.dumps(dict(metadata, fingerprint=fingerprint), indent=2))
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="" if smoke else str(gpu),
    )
    if (
        args.command in ("train", "smoke")
        and not (directory / "train.complete").exists()
    ):
        call_main(values, directory, "train", env)
        (directory / "train.complete").touch()
    if args.command not in ("eval", "smoke"):
        return
    checkpoint = directory / "checkpoints/last.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = file_hash(checkpoint)
    for evaluation_profile in profiles[job["profile"]].evaluations:
        evaluation_spec = profiles[evaluation_profile]
        evaluation_manifest = json.loads(
            (args.prepared / evaluation_profile / "manifest.json").read_text()
        )
        provenance = dict(
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_hash,
            training_fingerprint=source["fingerprint"],
            evaluation_fingerprint=evaluation_manifest["fingerprint"],
        )
        if (
            profiles[job["profile"]].task == "sudoku_shah"
            and evaluation_spec.task == "sudoku_extreme"
        ):
            audit_path = args.prepared / "shah_extreme_overlap.json"
            if not audit_path.exists():
                raise FileNotFoundError(
                    f"Run prepare or audit_reasoning_overlap.py first: {audit_path}"
                )
            audit = json.loads(audit_path.read_text())
            hashes = {
                entry["path"]: entry["sha256"]
                for entry in source["sources"] + evaluation_manifest["sources"]
            }
            if any(
                hashes.get(path) != digest for path, digest in audit["sources"].items()
            ):
                raise ValueError("OOD overlap audit does not match dataset sources")
            provenance["overlap_audit"] = audit
        # OOD only changes the compatible Sudoku data source, never model selection.
        for nfe in [4] if smoke else matrix.nfe:
            output = directory / "eval" / evaluation_profile / f"nfe_{nfe}"
            output.mkdir(parents=True, exist_ok=True)
            evaluation_metadata = output / "evaluation.json"
            expected = dict(
                provenance, requested_nfe=int(nfe), run_fingerprint=fingerprint
            )
            if (
                evaluation_metadata.exists()
                and json.loads(evaluation_metadata.read_text()) != expected
            ):
                raise ValueError(
                    f"evaluation checkpoint/data changed: {output}; use a new output root"
                )
            if (output / "eval.complete").exists() and evaluation_metadata.exists():
                continue
            evaluation_metadata.write_text(json.dumps(expected, indent=2))
            evaluation_values = values + [
                f"data={evaluation_spec.task}",
                f"data.prepared_dir={args.prepared / evaluation_profile}",
                "mode=reasoning_eval",
                f"eval.checkpoint_path={checkpoint}",
                f"sampling.steps={nfe}",
                f"reasoning.output_dir={output}",
                f"hydra.run.dir={output}",
            ]
            if smoke:
                evaluation_values += ["eval.disable_ema=true"]
            call_main(evaluation_values, output, "eval", env)
            (output / "eval.complete").touch()
    print(f"completed {job}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=["plan", "prepare", "smoke", "train", "eval", "aggregate"]
    )
    p.add_argument(
        "--matrix", type=Path, default=ROOT / "configs/sweeps/sudoku_rank.yaml"
    )
    p.add_argument(
        "--task-manifest",
        type=Path,
        default=ROOT / "configs/sweeps/reasoning_tasks.yaml",
    )
    p.add_argument("--profiles", nargs="+")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--modes", nargs="+", choices=["pure", "hybrid"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--root", type=Path, default=ROOT / "outputs/reasoning_rank")
    p.add_argument("--prepared", type=Path, default=ROOT / "data_cache/reasoning")
    p.add_argument("--raw", type=Path, default=ROOT / "data_cache/reasoning_raw")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--smoke-workers", type=int, default=2)
    p.add_argument(
        "--limit", type=int, help="preparation only: explicitly labeled source subset"
    )
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    args.root = args.root.resolve()
    args.prepared = args.prepared.resolve()
    matrix = OmegaConf.load(args.matrix)
    profiles = OmegaConf.load(args.task_manifest).profiles
    if args.command == "smoke":
        args.root = args.root / "smoke"
        args.gpus = list(range(args.smoke_workers))
    if not args.gpus:
        p.error("at least one worker is required")
    if args.command == "aggregate":
        aggregate(args.root)
        return
    selected = list(jobs(matrix, profiles, args))
    if not selected:
        p.error("empty selection")
    if args.command == "plan" or (
        args.command in ("train", "eval", "prepare") and not args.execute
    ):
        print(
            json.dumps(
                dict(
                    training_runs=len(selected),
                    nfe=list(matrix.nfe),
                    jobs=[
                        dict(
                            j,
                            overrides=overrides(
                                matrix, profiles, j, args.root, args.prepared
                            )[1],
                        )
                        for j in selected
                    ],
                ),
                indent=2,
            )
        )
        return
    if args.command == "prepare":
        required = set(j["profile"] for j in selected)
        required.update(
            e for name in list(required) for e in profiles[name].evaluations
        )
        for name in sorted(required):
            spec = profiles[name]
            if (args.prepared / name / "manifest.json").exists():
                existing = json.loads(
                    (args.prepared / name / "manifest.json").read_text()
                )
                expected_full = (
                    bool(spec.get("full_training", False)) and args.limit is None
                )
                if (
                    existing["task"] != spec.task
                    or existing["source_limit"] != args.limit
                    or existing["full_training"] != expected_full
                    or existing["prompt_length"] != spec.prompt_length
                    or existing["length"] != spec.length
                    or (
                        spec.task == "zebra"
                        and existing.get("encoding_version", 1) != 2
                    )
                    or existing["layout"]
                    != (list(spec.layout) if "layout" in spec else None)
                ):
                    raise ValueError(
                        f"existing prepared profile differs: {name}; use a new --prepared root"
                    )
                print(f"already prepared: {name}")
                continue
            print(
                prepare(
                    args.raw,
                    args.prepared / name,
                    spec.task,
                    layout=list(spec.layout) if "layout" in spec else None,
                    limit=args.limit,
                    full_training=bool(spec.get("full_training", False))
                    and args.limit is None,
                )
            )
        if "sudoku_shah" in {profiles[name].task for name in required}:
            from audit_reasoning_overlap import audit

            (args.prepared / "shah_extreme_overlap.json").write_text(
                json.dumps(audit(args.raw), indent=2)
            )
        return

    def queue(i):
        errors = []
        for job in selected[i :: len(args.gpus)]:
            try:
                run(matrix, profiles, job, args, args.gpus[i])
            except Exception as error:
                errors.append(str(error))
                print(str(error), file=sys.stderr, flush=True)
        return errors

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        failures = list(
            itertools.chain.from_iterable(pool.map(queue, range(len(args.gpus))))
        )
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; inspect logs above")
    if args.command in ("eval", "smoke"):
        aggregate(args.root)


if __name__ == "__main__":
    main()
