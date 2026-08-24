#!/usr/bin/env python3
"""Reproducible positional-vs-feature spectrum study for text blur bases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.analyze_owt_dct_power import dct_matrix, packed_token_sequences
except ModuleNotFoundError:
    from analyze_owt_dct_power import dct_matrix, packed_token_sequences


MODELS = (
    "openai-community/gpt2",
    "gpt2-large",
    "Qwen/Qwen2.5-0.5B",
)
MODEL_LABELS = {
    "openai-community/gpt2": "GPT-2 small",
    "gpt2-large": "GPT-2 large",
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
}


def spectrum_metrics(power: np.ndarray) -> dict[str, float | int]:
    power = np.maximum(np.asarray(power, dtype=np.float64), 0)
    fraction = power / power.sum()
    cumulative = np.cumsum(fraction)
    n = len(power)
    result: dict[str, float | int] = {
        "dimension": n,
        "participation_ratio": float(power.sum() ** 2 / np.square(power).sum()),
        "participation_ratio_fraction": float(
            power.sum() ** 2 / np.square(power).sum() / n
        ),
    }
    for percent in (1, 5, 10):
        count = max(1, math.ceil(n * percent / 100))
        result[f"top_{percent}pct_power"] = float(fraction[:count].sum())
    for target in (50, 90):
        result[f"modes_to_{target}pct"] = int(
            np.searchsorted(cumulative, target / 100) + 1
        )
        result[f"fraction_modes_to_{target}pct"] = float(
            result[f"modes_to_{target}pct"] / n
        )
    return result


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denominator) if denominator > 0 else float("nan")


@torch.inference_mode()
def feature_spectra(
    weight: torch.Tensor, counts: torch.Tensor, seed: int, initializer_std: float
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Uncentered, vocabulary-centered, weighted, and random spectra."""
    device = weight.device
    e = weight.double()
    vocab = len(e)
    frequencies = counts.to(device=device, dtype=torch.float64)
    frequencies = frequencies / frequencies.sum()
    vocabulary_mean = e.mean(0)
    weighted_mean = frequencies @ e
    centered = e - vocabulary_mean
    weighted_centered = e - weighted_mean
    covariances = {
        "uncentered": e.T @ e / vocab,
        "vocab_centered": centered.T @ centered / vocab,
        "frequency_weighted": (weighted_centered * frequencies[:, None]).T
        @ weighted_centered,
    }
    spectra: dict[str, np.ndarray] = {}
    top_vectors = None
    for name, covariance in covariances.items():
        values, vectors = torch.linalg.eigh(covariance)
        values, vectors = values.flip(0), vectors.flip(1)
        spectra[name] = values.clamp_min(0).cpu().numpy()
        if name == "frequency_weighted":
            top_vectors = vectors[:, :10]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_weight = (
        torch.empty(weight.shape, dtype=weight.dtype)
        .normal_(0, initializer_std, generator=generator)
        .to(device=device, dtype=torch.float64)
    )
    random_mean = frequencies @ random_weight
    random_vocabulary_mean = random_weight.mean(0)
    random_vocabulary_centered = random_weight - random_vocabulary_mean
    spectra["random_vocab_centered"] = (
        torch.linalg.eigvalsh(
            random_vocabulary_centered.T @ random_vocabulary_centered / vocab
        )
        .flip(0)
        .clamp_min(0)
        .cpu()
        .numpy()
    )
    random_covariance = ((random_weight - random_mean) * frequencies[:, None]).T @ (
        random_weight - random_mean
    )
    spectra["random_frequency_weighted"] = (
        torch.linalg.eigvalsh(random_covariance).flip(0).clamp_min(0).cpu().numpy()
    )
    learned_fraction = (
        spectra["frequency_weighted"] / spectra["frequency_weighted"].sum()
    )
    random_fraction = (
        spectra["random_frequency_weighted"]
        / spectra["random_frequency_weighted"].sum()
    )
    spectra["learned_over_random_norm_corrected"] = learned_fraction / np.maximum(
        random_fraction, 1e-300
    )
    log_frequency = torch.log1p(counts.to(device=device, dtype=torch.float64))
    projections = (e - weighted_mean) @ top_vectors
    correlations = [
        {
            "eigenvector": index,
            "pearson_log1p_frequency": pearson(projections[:, index], log_frequency),
            "eigenvalue": float(spectra["frequency_weighted"][index]),
        }
        for index in range(top_vectors.shape[1])
    ]
    return spectra, correlations


@torch.inference_mode()
def positional_spectra(
    sequences: list[list[int]],
    embedding: torch.nn.Embedding,
    random_weight: torch.Tensor,
    batch_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    device = embedding.weight.device
    length = len(sequences[0])
    basis = dct_matrix(length, device=device, dtype=embedding.weight.dtype)
    ids_all = torch.as_tensor(sequences, dtype=torch.long)
    counts = torch.bincount(ids_all.flatten(), minlength=embedding.num_embeddings)
    weights = counts.to(device=device, dtype=torch.float64)
    learned_centroid = (weights @ embedding.weight.double() / weights.sum()).to(
        embedding.weight.dtype
    )
    random_centroid = (weights @ random_weight.double() / weights.sum()).to(
        random_weight.dtype
    )
    learned = torch.zeros(length, dtype=torch.float64)
    shuffled = torch.zeros_like(learned)
    random = torch.zeros_like(learned)
    generator = torch.Generator(device=device).manual_seed(seed)
    for offset in range(0, len(sequences), batch_size):
        ids = ids_all[offset : offset + batch_size].to(device)
        x = embedding(ids) - learned_centroid
        coefficients = torch.einsum("lk,bld->bkd", basis, x)
        learned += coefficients.float().square().sum((0, 2)).double().cpu()
        order = torch.rand(x.shape[:2], device=device, generator=generator).argsort(1)
        shuffled_x = x.gather(1, order[:, :, None].expand_as(x))
        shuffled_coefficients = torch.einsum("lk,bld->bkd", basis, shuffled_x)
        shuffled += shuffled_coefficients.float().square().sum((0, 2)).double().cpu()
        random_x = torch.nn.functional.embedding(ids, random_weight) - random_centroid
        random_coefficients = torch.einsum("lk,bld->bkd", basis, random_x)
        random += random_coefficients.float().square().sum((0, 2)).double().cpu()
    result = {
        "learned": (learned / len(sequences)).numpy(),
        "shuffled": (shuffled / len(sequences)).numpy(),
        "random": (random / len(sequences)).numpy(),
    }
    lf = result["learned"] / result["learned"].sum()
    rf = result["random"] / result["random"].sum()
    result["learned_over_random_norm_corrected"] = lf / np.maximum(rf, 1e-300)
    return result


def load_gpt2_sequences(path: Path, n: int, length: int) -> list[list[int]]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    return [row[:length] for row in dataset.select(range(n))["input_ids"]]


def load_raw_texts(arrow_path: Path, limit: int = 5000):
    from datasets import Dataset

    dataset = Dataset.from_file(str(arrow_path))
    for text in dataset.select(range(min(limit, len(dataset))))["text"]:
        yield text


def tokenized_sequences_for_model(
    model_name: str,
    tokenizer,
    gpt2_sequences: list[list[int]],
    raw_arrow: Path,
    n: int,
    length: int,
) -> list[list[int]]:
    if model_name in {"openai-community/gpt2", "gpt2-large"}:
        return [row[:length] for row in gpt2_sequences[:n]]
    return list(packed_token_sequences(load_raw_texts(raw_arrow), tokenizer, length, n))


def line_svg(
    path: Path,
    title: str,
    ylabel: str,
    series: list[tuple[str, np.ndarray, str]],
    *,
    cumulative: bool,
    log_y: bool,
) -> None:
    width, height, left, right, top, bottom = 1100, 650, 95, 35, 70, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    processed = []
    for label, values, color in series:
        values = np.asarray(values, dtype=float)
        values = values / values.sum()
        if cumulative:
            values = np.cumsum(values)
        x = np.linspace(0, 1, len(values))
        processed.append((label, x, values, color))
    all_y = np.concatenate([item[2] for item in processed])
    ymin = max(float(all_y[all_y > 0].min()) * 0.75, 1e-12) if log_y else 0
    ymax = float(all_y.max()) * (1.1 if not cumulative else 1.0)

    def px(x):
        return left + plot_w * x

    def py(y):
        if log_y:
            y = np.log10(y)
            lo, hi = math.log10(ymin), math.log10(ymax)
            return top + plot_h * (1 - (y - lo) / (hi - lo))
        return top + plot_h * (1 - (y - ymin) / (ymax - ymin))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:23px;font-weight:600}"
        ".tick{font-size:13px}.label{font-size:16px}</style>",
        f'<text class="title" x="{width / 2}" y="35" text-anchor="middle">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#444"/>',
    ]
    if cumulative:
        svg.append(
            f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" '
            'stroke="#777" stroke-dasharray="7,5" stroke-width="2"/>'
        )
    for label, x, y, color in processed:
        points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px(x), py(y), strict=True))
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
    for tick in np.linspace(0, 1, 5):
        svg.append(
            f'<text class="tick" x="{px(tick)}" y="{top + plot_h + 22}" '
            f'text-anchor="middle">{tick:.2f}</text>'
        )
    for index, (label, _, _, color) in enumerate(processed):
        svg.append(
            f'<text class="label" x="{left + 15 + index * 210}" y="{top + 24}" '
            f'fill="{color}">{label}</text>'
        )
    svg.extend(
        [
            f'<text class="label" x="{width / 2}" y="{height - 18}" text-anchor="middle">normalized mode index</text>',
            f'<text class="label" x="22" y="{height / 2}" text-anchor="middle" '
            f'transform="rotate(-90 22 {height / 2})">{ylabel}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg) + "\n")


def layer_summary_svg(path: Path, metric_rows: list[dict]) -> None:
    """Plot normalized effective dimension against GPT-2 layer."""
    definitions = (
        ("hidden_positional", "real", "positional", "#276FBF"),
        ("hidden_positional", "shuffled", "positional shuffled", "#E07A5F"),
        ("hidden_feature", "real", "feature", "#6A4C93"),
    )
    series = []
    for domain, kind, label, color in definitions:
        selected = [
            row
            for row in metric_rows
            if row["domain"] == domain
            and str(row["configuration"]).endswith(f"_{kind}")
        ]
        selected.sort(key=lambda row: int(str(row["configuration"]).split("_")[1]))
        series.append(
            (
                label,
                np.asarray([row["participation_ratio_fraction"] for row in selected]),
                color,
            )
        )
    width, height, left, right, top, bottom = 1100, 650, 90, 35, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    layers = np.arange(len(series[0][1]))

    def px(value):
        return left + plot_w * value / layers[-1]

    def py(value):
        return top + plot_h * (1 - value)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:23px;font-weight:600}"
        ".tick{font-size:13px}.label{font-size:16px}</style>",
        '<text class="title" x="550" y="35" text-anchor="middle">GPT-2 hidden-state spectral concentration by layer</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#444"/>',
    ]
    for tick in np.linspace(0, 1, 6):
        svg.append(
            f'<line x1="{left}" y1="{py(tick):.2f}" x2="{width - right}" '
            f'y2="{py(tick):.2f}" stroke="#ddd"/>'
        )
        svg.append(
            f'<text class="tick" x="{left - 10}" y="{py(tick) + 4:.2f}" '
            f'text-anchor="end">{tick:.1f}</text>'
        )
    for layer in layers:
        svg.append(
            f'<text class="tick" x="{px(layer):.2f}" y="{top + plot_h + 22}" '
            f'text-anchor="middle">{layer}</text>'
        )
    for index, (label, values, color) in enumerate(series):
        points = " ".join(
            f"{px(layer):.2f},{py(value):.2f}"
            for layer, value in zip(layers, values, strict=True)
        )
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text class="label" x="{left + 15 + index * 230}" y="{top + 25}" '
            f'fill="{color}">{label}</text>'
        )
    svg.extend(
        [
            '<text class="label" x="550" y="630" text-anchor="middle">layer (0 = embeddings + positions)</text>',
            '<text class="label" x="20" y="325" text-anchor="middle" '
            'transform="rotate(-90 20 325)">participation ratio / dimension</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg) + "\n")


@torch.inference_mode()
def hidden_sweep(model, sequences, batch_size: int, device: torch.device, seed: int):
    """Feature covariance and real/shuffled positional power for every layer."""
    length = len(sequences[0])
    dimension = model.config.n_embd
    layers = model.config.n_layer + 1
    basis = dct_matrix(length, device=device, dtype=torch.float32)
    sums = torch.zeros(layers, dimension, dtype=torch.float64, device=device)
    cross = torch.zeros(
        layers, dimension, dimension, dtype=torch.float64, device=device
    )
    pos = torch.zeros(layers, length, dtype=torch.float64, device=device)
    shuffled = torch.zeros_like(pos)
    generator = torch.Generator(device=device).manual_seed(seed)
    token_count = 0
    for offset in range(0, len(sequences), batch_size):
        ids = torch.as_tensor(sequences[offset : offset + batch_size], device=device)
        states = model(
            input_ids=ids, output_hidden_states=True, use_cache=False
        ).hidden_states
        for layer, state in enumerate(states):
            state64 = state.double()
            sums[layer] += state64.sum((0, 1))
            cross[layer] += torch.einsum("bld,ble->de", state64, state64)
            coefficients = torch.einsum("lk,bld->bkd", basis, state.float())
            pos[layer] += coefficients.double().square().sum((0, 2))
            order = torch.rand(
                state.shape[:2], device=device, generator=generator
            ).argsort(1)
            permuted = state.gather(1, order[:, :, None].expand_as(state))
            shuffled_coefficients = torch.einsum("lk,bld->bkd", basis, permuted.float())
            shuffled[layer] += shuffled_coefficients.double().square().sum((0, 2))
        token_count += ids.numel()
    means = sums / token_count
    feature = []
    positional = []
    shuffled_positional = []
    for layer in range(layers):
        covariance = cross[layer] / token_count - torch.outer(
            means[layer], means[layer]
        )
        feature.append(
            torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0).cpu().numpy()
        )
        corrected = pos[layer] / len(sequences)
        corrected[0] -= length * means[layer].square().sum()
        positional.append(corrected.clamp_min(0).cpu().numpy())
        corrected_shuffle = shuffled[layer] / len(sequences)
        corrected_shuffle[0] -= length * means[layer].square().sum()
        shuffled_positional.append(corrected_shuffle.clamp_min(0).cpu().numpy())
    return feature, positional, shuffled_positional


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--hidden-sequences", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/blur_basis_study")
    )
    parser.add_argument(
        "--gpt2-data",
        type=Path,
        default=Path(
            "/home/patrick/.cache/discrete_diffusion/owt/"
            "openwebtext-train_train_bs1024_wrapped.dat"
        ),
    )
    parser.add_argument(
        "--raw-arrow",
        type=Path,
        default=Path(
            "/home/patrick/.cache/discrete_diffusion/owt/openwebtext/plain_text/0.0.0/"
            "b4325f019c648b1641a1784748667e8b74e5e064/openwebtext-train-00001-of-00080.arrow"
        ),
    )
    return parser.parse_args()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gpt2_sequences = load_gpt2_sequences(args.gpt2_data, args.num_sequences, 1024)
    metric_rows, correlation_rows = [], []
    saved_spectra = {}
    revisions = {}
    for model_index, model_name in enumerate(MODELS):
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_name, local_files_only=True, torch_dtype=torch.float32
            )
            .to(device)
            .eval()
        )
        revisions[model_name] = getattr(model.config, "_commit_hash", None)
        embedding = model.get_input_embeddings()
        sequences = tokenized_sequences_for_model(
            model_name,
            tokenizer,
            gpt2_sequences,
            args.raw_arrow,
            args.num_sequences,
            1024,
        )
        ids = torch.as_tensor(sequences, dtype=torch.long)
        counts = torch.bincount(ids.flatten(), minlength=embedding.num_embeddings)
        initializer_std = float(getattr(model.config, "initializer_range", 0.02))
        feature, correlations = feature_spectra(
            embedding.weight, counts, args.seed + model_index, initializer_std
        )
        saved_spectra[f"{model_name}/feature"] = feature
        for kind in (
            "uncentered",
            "vocab_centered",
            "frequency_weighted",
            "random_vocab_centered",
            "random_frequency_weighted",
        ):
            metric_rows.append(
                {
                    "model": MODEL_LABELS[model_name],
                    "domain": "feature",
                    "configuration": kind,
                    **spectrum_metrics(feature[kind]),
                }
            )
        for row in correlations:
            correlation_rows.append({"model": MODEL_LABELS[model_name], **row})
        if model_name == "openai-community/gpt2":
            generator = torch.Generator().manual_seed(args.seed)
            random_weight = (
                torch.empty_like(embedding.weight, device="cpu")
                .normal_(0, initializer_std, generator=generator)
                .to(device)
            )
            for length in (128, 256, 512, 1024):
                positional = positional_spectra(
                    [row[:length] for row in sequences],
                    embedding,
                    random_weight,
                    args.batch_size,
                    args.seed,
                )
                saved_spectra[f"{model_name}/positional_L{length}"] = positional
                for kind in ("learned", "shuffled", "random"):
                    metric_rows.append(
                        {
                            "model": MODEL_LABELS[model_name],
                            "domain": "positional",
                            "configuration": f"L{length}_{kind}",
                            **spectrum_metrics(positional[kind]),
                        }
                    )
            headline_pos = saved_spectra[f"{model_name}/positional_L1024"]["learned"]
            line_svg(
                args.output_dir / "headline_cumulative.svg",
                "Positional vs feature concentration — GPT-2 small",
                "cumulative power fraction",
                [
                    ("positional DCT", headline_pos, "#276FBF"),
                    ("feature covariance", feature["frequency_weighted"], "#D1495B"),
                ],
                cumulative=True,
                log_y=False,
            )
            line_svg(
                args.output_dir / "positional_detail.svg",
                "Positional spectra — GPT-2 small, L=1024",
                "fraction of total power",
                [
                    ("real", headline_pos, "#276FBF"),
                    (
                        "shuffled",
                        saved_spectra[f"{model_name}/positional_L1024"]["shuffled"],
                        "#E07A5F",
                    ),
                    (
                        "random table",
                        saved_spectra[f"{model_name}/positional_L1024"]["random"],
                        "#777777",
                    ),
                ],
                cumulative=False,
                log_y=True,
            )
            line_svg(
                args.output_dir / "feature_detail.svg",
                "Feature spectra — GPT-2 small",
                "fraction of total power",
                [
                    ("uncentered", feature["uncentered"], "#E07A5F"),
                    ("centered", feature["vocab_centered"], "#276FBF"),
                    ("frequency weighted", feature["frequency_weighted"], "#D1495B"),
                    (
                        "random weighted",
                        feature["random_frequency_weighted"],
                        "#777777",
                    ),
                ],
                cumulative=False,
                log_y=True,
            )
            hidden_sequences = sequences[: args.hidden_sequences]
            hidden_feature, hidden_pos, hidden_shuffle = hidden_sweep(
                model, hidden_sequences, args.hidden_batch_size, device, args.seed
            )
            for layer, (fs, ps, ss) in enumerate(
                zip(hidden_feature, hidden_pos, hidden_shuffle, strict=True)
            ):
                for domain, kind, spectrum in (
                    ("hidden_feature", "real", fs),
                    ("hidden_positional", "real", ps),
                    ("hidden_positional", "shuffled", ss),
                ):
                    metric_rows.append(
                        {
                            "model": MODEL_LABELS[model_name],
                            "domain": domain,
                            "configuration": f"layer_{layer}_{kind}",
                            **spectrum_metrics(spectrum),
                        }
                    )
            saved_spectra[f"{model_name}/hidden_feature"] = hidden_feature
            saved_spectra[f"{model_name}/hidden_positional"] = hidden_pos
            saved_spectra[f"{model_name}/hidden_positional_shuffled"] = hidden_shuffle
        del model
        torch.cuda.empty_cache()
    with (args.output_dir / "summary_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (args.output_dir / "frequency_correlations.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlation_rows[0]))
        writer.writeheader()
        writer.writerows(correlation_rows)
    layer_summary_svg(args.output_dir / "hidden_layer_sweep.svg", metric_rows)
    line_svg(
        args.output_dir / "feature_models_cumulative.svg",
        "Frequency-weighted feature concentration across models",
        "cumulative power fraction",
        [
            (
                MODEL_LABELS[name],
                saved_spectra[f"{name}/feature"]["frequency_weighted"],
                color,
            )
            for name, color in zip(
                MODELS, ("#276FBF", "#D1495B", "#6A4C93"), strict=True
            )
        ],
        cumulative=True,
        log_y=False,
    )
    flat = {}
    for prefix, group in saved_spectra.items():
        if isinstance(group, dict):
            for name, values in group.items():
                flat[f"{prefix}/{name}"] = np.asarray(values)
        elif isinstance(group, list):
            for index, values in enumerate(group):
                flat[f"{prefix}/layer_{index}"] = np.asarray(values)
    np.savez_compressed(args.output_dir / "spectra.npz", **flat)
    metadata = {
        "models": list(MODELS),
        "model_revisions": revisions,
        "dataset": "openwebtext cached Arrow + prepared GPT-2 wrapped sequences",
        "gpt2_data": str(args.gpt2_data),
        "raw_arrow": str(args.raw_arrow),
        "num_sequences": args.num_sequences,
        "hidden_sequences": args.hidden_sequences,
        "seed": args.seed,
        "torch": torch.__version__,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote full study to {args.output_dir}")


if __name__ == "__main__":
    main()
