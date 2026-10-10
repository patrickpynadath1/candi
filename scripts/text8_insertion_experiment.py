"""Run the matched 22M Text8 experiments and save unselected sample reports.

Use the repository's Python environment and select an idle GPU with
CUDA_VISIBLE_DEVICES. --wait-for-pid attaches the evaluation pipeline to an
already running insertion training process; it does not start a second copy.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def summarize(path):
    result = json.loads(path.read_text())
    records = result["records"]
    lengths = result.get("generated_lengths", [256] * len(records))
    return {
        "checkpoint_step": result["checkpoint_step"],
        "seed": result["seed"],
        "samples": len(records),
        "valid_word_rate": result["valid_word_rate"],
        "generated_words": result["generated"],
        "held_out_words": result["held_out"],
        "length_mean": statistics.mean(lengths),
        "length_stdev": statistics.pstdev(lengths),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "empty_samples": lengths.count(0),
        "truncated_samples": sum(result.get("insertion_truncated", [])),
        "special_tokens": sum(r["generated"]["special_token_count"] for r in records),
        "unique_sequences": len({r["generated"]["text"] for r in records}),
        "results": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    report = ROOT / "core/outputs/text8_insertion_comparison"
    report.mkdir(parents=True, exist_ok=True)
    status = report / "status.json"
    summaries = {}

    def mark(stage, **extra):
        value = {"stage": stage, "updated_at_unix": time.time(), **extra}
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(status)
        print(json.dumps(value), flush=True)

    def run(experiment, output, *overrides):
        if (output / "config.yaml").exists():
            raise FileExistsError(f"refusing to overwrite existing run: {output}")
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "main.py",
            "+experiment=" + experiment,
            "output_dir=" + str(output),
            "data.cache_dir=" + args.cache_dir,
            *overrides,
        ]
        with (output / "pipeline.log").open("w") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    try:
        for experiment in ("text8_insertion_22m", "text8_hybrid_22m"):
            output = ROOT / "core/outputs" / experiment
            mark("training", experiment=experiment)
            if experiment == "text8_insertion_22m" and args.wait_for_pid:
                # Check identity as well as PID; never wait on a reused PID.
                proc = Path("/proc") / str(args.wait_for_pid) / "cmdline"
                while proc.exists():
                    try:
                        command = proc.read_bytes()
                    except FileNotFoundError:
                        break
                    if (
                        b"main.py" not in command
                        or b"+experiment=text8_insertion_22m" not in command
                    ):
                        break
                    time.sleep(15)
            else:
                run(experiment, output)
            checkpoint = output / "checkpoints/last.ckpt"
            saved = torch.load(
                checkpoint, map_location="cpu", weights_only=False, mmap=True
            )
            step = saved["global_step"]
            if step != 5000:
                raise RuntimeError(
                    f"{experiment}: expected 5000 completed steps, got {step}"
                )
            del saved
            for seed in (23, 24):
                label = f"{experiment}_test_seed{seed}"
                evaluation = output / f"test_seed{seed}"
                mark("evaluating", experiment=experiment, seed=seed)
                run(
                    experiment,
                    evaluation,
                    "mode=evaluate",
                    "evaluation.checkpoint=" + str(checkpoint),
                    "task.evaluation_split=test",
                    "seed=" + str(seed),
                )
                summaries[label] = summarize(evaluation / "results.json")
                (report / "summary.json").write_text(json.dumps(summaries, indent=2))
                result = json.loads((evaluation / "results.json").read_text())
                with (report / f"{label}_samples.txt").open("w") as stream:
                    for i, row in enumerate(result["records"]):
                        stream.write(f"[{i}] {row['generated']['text']}\n\n")
            metrics = output / "metrics/version_0/metrics.csv"
            with metrics.open() as stream:
                rows = [
                    {k: v for k, v in row.items() if v}
                    for row in csv.DictReader(stream)
                ]
            (report / f"{experiment}_losses.json").write_text(
                json.dumps(
                    [r for r in rows if "train/loss" in r or "val/nll" in r], indent=2
                )
            )
        mark("complete", summary=str(report / "summary.json"))
    except Exception as exc:
        mark("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
