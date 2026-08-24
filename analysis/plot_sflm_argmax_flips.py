#!/usr/bin/env python3
"""Plot mean argmax flips over ancestral TinyGSM sampling time."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_traces(path: Path) -> np.ndarray:
    with path.open() as handle:
        payload = json.load(handle)
    traces = np.asarray(
        [record["argmax_changes_per_step"] for record in payload["records"]],
        dtype=float,
    )
    if traces.ndim != 2 or traces.shape[0] == 0:
        raise ValueError(f"expected a nonempty rectangular trace matrix in {path}")
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm", type=Path, required=True)
    parser.add_argument("--duo", type=Path, required=True)
    parser.add_argument("--candi-blur", type=Path)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    series = {"MDLM": load_traces(args.mdlm), "Duo": load_traces(args.duo)}
    if args.candi_blur is not None:
        series["CANDI blur"] = load_traces(args.candi_blur)
    lengths = {values.shape[1] for values in series.values()}
    if len(lengths) != 1:
        raise ValueError("MDLM and Duo traces have different numbers of comparisons")
    comparisons = lengths.pop()
    # A comparison is recorded after each new clean prediction. The first
    # comparison follows the second NFE; the last is the t≈0 noise-removal NFE.
    diffusion_time = np.linspace(1.0, 0.0, comparisons + 1)[1:]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "comparison", "diffusion_time", "mean_flips", "sem_flips"])
        for name, values in series.items():
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            for index, (time, avg, error) in enumerate(zip(diffusion_time, mean, sem), 1):
                writer.writerow([name, index, time, avg, error])

    num_axes = 1 if args.overlay else len(series)
    fig, axes = plt.subplots(
        num_axes, 1, figsize=(9, 5.5 if args.overlay else 3.25 * len(series) + 0.5),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = {"MDLM": "#3366cc", "Duo": "#dc3912", "CANDI blur": "#109618"}
    for index, (name, values) in enumerate(series.items()):
        axis = axes[0] if args.overlay else axes[index]
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        color = colors[name]
        axis.plot(diffusion_time, mean, color=color, linewidth=2, label=name)
        axis.fill_between(diffusion_time, mean - sem, mean + sem, color=color, alpha=0.2)
        axis.set_ylabel("Mean argmax flips")
        if not args.overlay:
            axis.set_title(f"{name} (n={values.shape[0]})")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("Diffusion time (noisy → clean)")
    axes[-1].invert_xaxis()
    fig.suptitle("TinyGSM ancestral sampling: argmax decision changes", fontsize=14)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
