#!/usr/bin/env python3
"""Plot the corrected Text8 bandwidth sweep, including high-sigma models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


SIGMAS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
NFES = (32, 64, 128)
METRICS = ("total_valid_words_5", "total_valid_words_6")
BUNDLE_ROOT = Path(__file__).resolve().parent.parent


def sigma_label(sigma: float) -> str:
    prefix = "old" if sigma <= 2.5 else "new"
    if prefix == "old":
        value = f"{sigma:.1f}".replace(".", "p")
    else:
        value = str(int(sigma))
    return f"{prefix}_sigma_{value}"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def best_at_nfe(data: dict, nfe: int, metric: str) -> tuple[float, float]:
    return max(
        (float(values[metric]), float(temperature))
        for temperature, values in data[str(nfe)].items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=BUNDLE_ROOT / "data",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=BUNDLE_ROOT / "reference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BUNDLE_ROOT / "outputs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_data = {}
    for sigma in SIGMAS:
        path = args.sweep_dir / f"{sigma_label(sigma)}.json"
        sweep_data[sigma] = load_json(path)

    candi = load_json(args.reference_dir / "candi_base.json")
    mdlm = load_json(args.reference_dir / "mdlm.json")

    rows = []
    for nfe in NFES:
        for sigma in SIGMAS:
            for metric in METRICS:
                value, temperature = best_at_nfe(sweep_data[sigma], nfe, metric)
                rows.append(
                    {
                        "nfe": nfe,
                        "sigma": sigma,
                        "metric": metric,
                        "best_value": value,
                        "best_temperature": temperature,
                    }
                )

    csv_path = args.output_dir / "text8_extended_sigma_best_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    navy = "#173f63"
    red = "#a44232"
    gray = "#aaa8a3"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), sharey=True)

    for index, (ax, nfe) in enumerate(zip(axes, NFES)):
        for metric, linestyle, marker in (
            ("total_valid_words_5", "-", "o"),
            ("total_valid_words_6", "--", "D"),
        ):
            values = [best_at_nfe(sweep_data[s], nfe, metric)[0] for s in SIGMAS]
            ax.plot(
                SIGMAS,
                values,
                color=navy,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                linewidth=2.4,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )

            candi_value = best_at_nfe(candi, nfe, metric)[0]
            mdlm_value = best_at_nfe(mdlm, nfe, metric)[0]
            ax.axhline(
                candi_value,
                color=red,
                linestyle=linestyle,
                linewidth=1.8,
                alpha=0.9,
                zorder=1,
            )
            ax.axhline(
                mdlm_value,
                color=gray,
                linestyle=linestyle,
                linewidth=1.8,
                alpha=0.9,
                zorder=1,
            )

        ax.axvline(2.75, color="#777777", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(
            2.85,
            0.204,
            "new high-$\\sigma_W$",
            color="#666666",
            fontsize=8,
            va="top",
        )
        ax.set_title(f"NFE = {nfe}", loc="left", fontsize=16)
        ax.set_xticks(SIGMAS)
        ax.set_xticklabels(
            ["0.5", "1", "1.5", "2", "2.5", "3", "4", "6", "8"],
            rotation=35,
        )
        ax.set_xlim(0.25, 8.25)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"Blending bandwidth $\sigma_W$", fontsize=13)
        if index:
            ax.spines["left"].set_alpha(0.45)

    axes[0].set_ylabel("Valid words", fontsize=15)
    axes[0].set_ylim(0.06, 0.21)

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], color=navy, marker="o", linewidth=2.4, label="SJD ≥5"),
        Line2D(
            [0], [0], color=navy, marker="D", linestyle="--", linewidth=2.4,
            label="SJD ≥6",
        ),
        Line2D([0], [0], color=red, linewidth=2.0, label="CANDI"),
        Line2D([0], [0], color=gray, linewidth=2.0, label="MDLM"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png_path = args.output_dir / "text8_extended_sigma_sweep.png"
    pdf_path = args.output_dir / "text8_extended_sigma_sweep.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
