#!/usr/bin/env python3
"""Aggregate persisted text8 metrics and mark temperature Pareto frontiers."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "new_results"


def is_frontier(point, peers):
    diversity = point["total_unique_words"]
    coherence = point["total_valid_words"]
    return not any(
        other is not point
        and other["total_unique_words"] >= diversity
        and other["total_valid_words"] >= coherence
        and (
            other["total_unique_words"] > diversity
            or other["total_valid_words"] > coherence
        )
        for other in peers
    )


def main():
    rows = []
    for path in sorted(RESULTS.glob("*/*.json")):
        with path.open() as handle:
            record = json.load(handle)
        metrics = record["metrics"]
        rows.append(
            {
                "algorithm": record["algorithm"],
                "nfe": record["steps"],
                "temperature": record["temperature"],
                "num_samples": record["num_samples"],
                "total_unique_words": metrics["total_unique_words"],
                "total_valid_words": metrics["total_valid_words"],
                "source_file": str(path.relative_to(ROOT)),
            }
        )

    for row in rows:
        peers = [candidate for candidate in rows if candidate["algorithm"] == row["algorithm"] and candidate["nfe"] == row["nfe"]]
        row["pareto_frontier"] = is_frontier(row, peers)

    rows.sort(key=lambda row: (row["algorithm"], row["nfe"], row["temperature"]))
    output_json = RESULTS / "frontiers.json"
    output_csv = RESULTS / "frontiers.csv"
    output_json.write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} evaluations to {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
