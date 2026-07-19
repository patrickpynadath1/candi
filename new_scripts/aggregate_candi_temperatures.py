#!/usr/bin/env python3
"""Aggregate CANDI Sudoku temperature sweeps, including the temp=1 baseline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE_ROOT = ROOT / "new_candi_temperature_results"
BASELINE_ROOT = ROOT / "new_candi_curve_results"


def load_result(path, variant, temperature):
    result = json.loads(path.read_text())
    return {
        "variant": variant,
        "temperature": temperature,
        "accuracy": result["accuracy"],
        "valid_grid_rate": result["valid_grid_rate"],
        "cell_accuracy": result["cell_accuracy"],
        "num_correct": result["num_correct"],
        "num_total": result["num_total"],
    }


def main():
    rows = []
    variants = sorted(path.name for path in TEMPERATURE_ROOT.iterdir() if path.is_dir() and path.name != "logs")
    for variant in variants:
        for path in sorted((TEMPERATURE_ROOT / variant / "nfe_90").glob("temp_*/results.json")):
            temperature = float(path.parent.name.removeprefix("temp_"))
            rows.append(load_result(path, variant, temperature))
        baseline = BASELINE_ROOT / variant / "nfe_90" / "results.json"
        if baseline.exists():
            rows.append(load_result(baseline, variant, 1.0))

    rows.sort(key=lambda row: (row["variant"], row["temperature"]))
    (TEMPERATURE_ROOT / "summary.json").write_text(json.dumps(rows, indent=2))
    header = "variant\ttemperature\taccuracy\tvalid_grid_rate\tcell_accuracy\tcorrect\ttotal"
    lines = [header] + [
        f'{row["variant"]}\t{row["temperature"]:.2f}\t{row["accuracy"]:.6f}\t'
        f'{row["valid_grid_rate"]:.6f}\t{row["cell_accuracy"]:.6f}\t'
        f'{row["num_correct"]}\t{row["num_total"]}'
        for row in rows
    ]
    (TEMPERATURE_ROOT / "summary.tsv").write_text("\n".join(lines) + "\n")
    for variant in variants:
        candidates = [row for row in rows if row["variant"] == variant]
        best = max(candidates, key=lambda row: row["accuracy"])
        print(
            f'{variant}\tbest_temp={best["temperature"]:.2f}\t'
            f'accuracy={best["accuracy"]:.3f}\tvalid={best["valid_grid_rate"]:.3f}'
        )


if __name__ == "__main__":
    main()
