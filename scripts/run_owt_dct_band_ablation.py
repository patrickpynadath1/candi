#!/usr/bin/env python3
"""Run positional DCT band ablations through GPT-2 without token decoding.

For every octave band, this script removes the selected DCT coefficients from
the corpus-centered GPT-2 token embeddings, reconstructs positional embeddings,
and measures causal language-model loss. Equal-power controls remove random
modes across the spectrum whose empirical power sums as closely as possible to
the band's power.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scripts.analyze_owt_dct_power import corpus_centroid, dct_matrix
except (
    ModuleNotFoundError
):  # Direct execution puts scripts/, not repo root, on sys.path.
    from analyze_owt_dct_power import corpus_centroid, dct_matrix


def octave_bands(length: int) -> list[tuple[int, int]]:
    """DC plus non-overlapping octave bands covering [0, length)."""
    if length < 2 or length & (length - 1):
        raise ValueError("sequence length must be a power of two and at least 2")
    bands = [(0, 1), (1, min(3, length))]
    start = min(3, length)
    while start < length:
        stop = min(2 * start - 1, length)
        bands.append((start, stop))
        start = stop
    return [band for band in bands if band[0] < band[1]]


def equal_power_subsets(
    mode_power: torch.Tensor,
    band: tuple[int, int],
    repeats: int,
    seed: int,
) -> list[torch.Tensor]:
    """Random spectrum-wide subsets whose power is closest to target power."""
    start, stop = band
    target = mode_power[start:stop].sum()
    candidates = torch.arange(len(mode_power))
    generator = torch.Generator().manual_seed(seed)
    subsets = []
    for _ in range(repeats):
        order = candidates[torch.randperm(len(candidates), generator=generator)]
        cumulative = mode_power[order].cumsum(0)
        crossing = int(torch.searchsorted(cumulative, target).clamp_max(len(order) - 1))
        choices = [crossing, crossing - 1]
        last = min(
            choices,
            key=lambda i: abs(float((cumulative[i] if i >= 0 else 0) - target)),
        )
        subsets.append(order[: last + 1])
    return subsets


@torch.inference_mode()
def empirical_mode_power(
    sequences: list[list[int]],
    embedding: torch.nn.Embedding,
    centroid: torch.Tensor,
    basis: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros(basis.shape[1], dtype=torch.float64)
    for offset in range(0, len(sequences), batch_size):
        ids = torch.as_tensor(sequences[offset : offset + batch_size], device=device)
        coefficients = torch.einsum("lk,bld->bkd", basis, embedding(ids) - centroid)
        total += coefficients.float().square().sum(dim=(0, 2)).double().cpu()
    return total / len(sequences)


def per_sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), labels[:, 1:], reduction="none"
    )
    return losses.mean(dim=1)


@torch.inference_mode()
def evaluate(
    model,
    embedding: torch.nn.Embedding,
    sequences: list[list[int]],
    centroid: torch.Tensor,
    basis: torch.Tensor,
    interventions: list[dict],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [intervention, sequence] losses, including baseline at row zero."""
    all_losses = torch.empty(len(interventions) + 1, len(sequences))
    for offset in range(0, len(sequences), batch_size):
        ids = torch.as_tensor(sequences[offset : offset + batch_size], device=device)
        x = embedding(ids)
        baseline_logits = model(inputs_embeds=x, use_cache=False).logits
        all_losses[0, offset : offset + len(ids)] = per_sequence_loss(
            baseline_logits, ids
        ).cpu()
        coefficients = torch.einsum("lk,bld->bkd", basis, x - centroid)
        for index, intervention in enumerate(interventions, start=1):
            corrupted = coefficients.clone()
            corrupted[:, intervention["modes"].to(device)] = 0
            reconstructed = torch.einsum("lk,bkd->bld", basis, corrupted) + centroid
            logits = model(inputs_embeds=reconstructed, use_cache=False).logits
            all_losses[index, offset : offset + len(ids)] = per_sequence_loss(
                logits, ids
            ).cpu()
    return all_losses[0], all_losses[1:]


def summarize(
    baseline: torch.Tensor, losses: torch.Tensor, interventions: list[dict]
) -> list[dict]:
    rows = []
    deltas = losses - baseline[None]
    for intervention, values, absolute in zip(
        interventions, deltas, losses, strict=True
    ):
        rows.append(
            {
                **{k: v for k, v in intervention.items() if k != "modes"},
                "num_modes": len(intervention["modes"]),
                "mean_loss": float(absolute.mean()),
                "mean_loss_increase": float(values.mean()),
                "stderr_loss_increase": float(
                    values.std(unbiased=True) / math.sqrt(len(values))
                )
                if len(values) > 1
                else 0.0,
            }
        )
    return rows


def save_plot(path: Path, rows: list[dict]) -> None:
    """Plot band effects and equal-power control mean/range as SVG."""
    band_rows = [row for row in rows if row["kind"] == "band"]
    controls = {
        label: [
            row for row in rows if row["kind"] == "equal_power" and row["band"] == label
        ]
        for label in (row["band"] for row in band_rows)
    }
    values = [row["mean_loss_increase"] for row in rows]
    ymin, ymax = min(0.0, min(values)), max(values) * 1.12
    width, height, left, right, top, bottom = 1200, 620, 90, 35, 65, 150
    plot_w, plot_h = width - left - right, height - top - bottom

    def py(value: float) -> float:
        return top + plot_h * (1 - (value - ymin) / (ymax - ymin))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:23px;font-weight:600}"
        ".tick{font-size:13px}.label{font-size:16px}</style>",
        '<text class="title" x="600" y="32" text-anchor="middle">GPT-2 loss after DCT band removal</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#444"/>',
    ]
    for value in np.linspace(ymin, ymax, 6):
        y = py(float(value))
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#ddd"/>'
        )
        svg.append(
            f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.3f}</text>'
        )
    step = plot_w / len(band_rows)
    points = []
    control_points = []
    for index, row in enumerate(band_rows):
        x = left + step * (index + 0.5)
        points.append(f"{x:.2f},{py(row['mean_loss_increase']):.2f}")
        control_values = [item["mean_loss_increase"] for item in controls[row["band"]]]
        control_mean = float(np.mean(control_values))
        control_points.append(f"{x:.2f},{py(control_mean):.2f}")
        svg.append(
            f'<line x1="{x:.2f}" y1="{py(min(control_values)):.2f}" x2="{x:.2f}" '
            f'y2="{py(max(control_values)):.2f}" stroke="#E07A5F" stroke-width="3"/>'
        )
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="end" '
            f'transform="rotate(-45 {x:.2f} {top + plot_h + 20})">{row["band"]}</text>'
        )
    svg.append(
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#276FBF" stroke-width="3"/>'
    )
    svg.append(
        f'<polyline points="{" ".join(control_points)}" fill="none" stroke="#E07A5F" stroke-width="2"/>'
    )
    for points_to_draw, color in [(points, "#276FBF"), (control_points, "#E07A5F")]:
        for point in points_to_draw:
            x, y = point.split(",")
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
    svg.extend(
        [
            f'<text class="label" x="{left + 15}" y="{top + 24}" fill="#276FBF">band</text>',
            f'<text class="label" x="{left + 70}" y="{top + 24}" fill="#E07A5F">equal-power random</text>',
            '<text class="label" x="22" y="270" text-anchor="middle" transform="rotate(-90 22 270)">loss increase (nats/token)</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/home/patrick/.cache/discrete_diffusion/owt/"
            "openwebtext-train_train_bs1024_wrapped.dat"
        ),
    )
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--control-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/owt_gpt2_band_ablation")
    )
    return parser.parse_args()


def main() -> None:
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM

    args = parse_args()
    device = torch.device(args.device)
    dataset = load_from_disk(str(args.dataset_path))
    sequences = dataset.select(range(min(args.num_sequences, len(dataset))))[
        "input_ids"
    ]
    length = len(sequences[0])
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    embedding = model.get_input_embeddings()
    centroid = corpus_centroid(sequences, embedding)
    basis = dct_matrix(length, device=device, dtype=embedding.weight.dtype)
    power = empirical_mode_power(
        sequences, embedding, centroid, basis, args.batch_size, device
    )
    interventions = []
    for band_index, band in enumerate(octave_bands(length)):
        start, stop = band
        label = "0" if stop == 1 else f"{start}-{stop - 1}"
        band_power = float(power[start:stop].sum())
        interventions.append(
            {
                "kind": "band",
                "band": label,
                "repeat": -1,
                "removed_power": band_power,
                "modes": torch.arange(start, stop),
            }
        )
        controls = equal_power_subsets(
            power, band, args.control_repeats, args.seed + 1009 * band_index
        )
        for repeat, modes in enumerate(controls):
            interventions.append(
                {
                    "kind": "equal_power",
                    "band": label,
                    "repeat": repeat,
                    "removed_power": float(power[modes].sum()),
                    "modes": modes,
                }
            )
    baseline, losses = evaluate(
        model,
        embedding,
        sequences,
        centroid,
        basis,
        interventions,
        args.batch_size,
        device,
    )
    rows = summarize(baseline, losses, interventions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "model": args.model,
        "dataset_path": str(args.dataset_path),
        "num_sequences": len(sequences),
        "sequence_length": length,
        "baseline_loss": float(baseline.mean()),
        "control_repeats": args.control_repeats,
        "seed": args.seed,
        "corpus_centered": True,
        "rows": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_plot(args.output.with_suffix(".svg"), rows)
    print(f"Baseline loss: {baseline.mean():.4f}; wrote {args.output}.{{csv,json,svg}}")


if __name__ == "__main__":
    main()
