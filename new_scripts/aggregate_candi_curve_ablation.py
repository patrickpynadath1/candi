#!/usr/bin/env python3
"""Aggregate completed CANDI curve screening evaluations."""

import csv
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
results_root = root / "new_candi_curve_results"
rows = []
for path in sorted(results_root.glob("*/nfe_*/results.json")):
    metrics = json.loads(path.read_text())
    rows.append(
        {
            "run": path.parents[1].name,
            "nfe": int(path.parent.name.removeprefix("nfe_")),
            "accuracy": metrics["accuracy"],
            "cell_accuracy": metrics["cell_accuracy"],
            "valid_grid_rate": metrics["valid_grid_rate"],
            "num_total": metrics["num_total"],
        }
    )
rows.sort(key=lambda row: row["accuracy"], reverse=True)
destination = results_root / "summary.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["run"])
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(
        f"{row['run']:36s} exact={row['accuracy']:.2%} "
        f"cell={row['cell_accuracy']:.2%} valid={row['valid_grid_rate']:.2%}"
    )
print(f"wrote {destination}")
