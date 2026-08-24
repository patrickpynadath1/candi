#!/usr/bin/env python3
"""Estimate positional DCT-mode power of GPT-2 token embeddings on OWT.

The script packs the OpenWebText token stream into fixed-length sequences,
looks up GPT-2's (token-only) embeddings, applies the same orthonormal DCT-II
used by ``algos.blurring_diffusion``, and averages each mode's squared L2
coefficient over sequences.

Example:
    uv run python scripts/analyze_owt_dct_power.py \
        --num-sequences 10000 --sequence-length 1024 \
        --output outputs/owt_gpt2_dct_power

By default, the script reuses the tokenized dataset produced by the repository
at ``<cache-dir>/openwebtext-train_train_bs<L>_wrapped.dat``. It only falls
back to streaming and tokenizing raw OWT when that prepared dataset is absent.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch


def dct_matrix(length: int, *, device: torch.device, dtype: torch.dtype):
    """Orthonormal DCT-II synthesis matrix V[n, k]."""
    n = torch.arange(length, device=device, dtype=dtype)[:, None]
    k = torch.arange(length, device=device, dtype=dtype)[None, :]
    matrix = torch.cos(math.pi * (n + 0.5) * k / length)
    scale = torch.full((length,), math.sqrt(2.0 / length), device=device, dtype=dtype)
    scale[0] = math.sqrt(1.0 / length)
    return matrix * scale


def packed_token_sequences(
    texts: Iterable[str], tokenizer, sequence_length: int, limit: int
) -> Iterator[list[int]]:
    """Pack documents into full sequences, inserting EOS at boundaries."""
    buffer: list[int] = []
    emitted = 0
    eos = tokenizer.eos_token_id
    for text in texts:
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids.append(eos)
        buffer.extend(ids)
        while len(buffer) >= sequence_length and emitted < limit:
            yield buffer[:sequence_length]
            del buffer[:sequence_length]
            emitted += 1
        if emitted >= limit:
            return


@torch.inference_mode()
def corpus_centroid(
    sequences: Sequence[Sequence[int]], embedding: torch.nn.Embedding
) -> torch.Tensor:
    """Token-weighted embedding mean over exactly the sequences being analyzed."""
    token_ids = torch.as_tensor(sequences, dtype=torch.long, device="cpu")
    counts = torch.bincount(token_ids.flatten(), minlength=embedding.num_embeddings).to(
        device=embedding.weight.device, dtype=torch.float64
    )
    # Accumulate in float64: this is cheap (one vector per vocabulary item) and
    # avoids making mode zero sensitive to summation error over many tokens.
    return (counts @ embedding.weight.double() / counts.sum()).to(
        dtype=embedding.weight.dtype
    )


@torch.inference_mode()
def accumulate_mode_power(
    sequences: Iterable[Sequence[int]],
    embedding: torch.nn.Embedding,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    centroid: torch.Tensor | None = None,
    random_embedding: torch.nn.Embedding | None = None,
    random_centroid: torch.Tensor | None = None,
    shuffle_baseline: bool = False,
    seed: int = 0,
) -> dict[str, torch.Tensor | int]:
    """Return per-mode first and second moments over sequences."""
    basis = dct_matrix(sequence_length, device=device, dtype=embedding.weight.dtype)
    sums = torch.zeros(sequence_length, dtype=torch.float64)
    sums_sq = torch.zeros_like(sums)
    shuffled_sums = torch.zeros_like(sums) if shuffle_baseline else None
    random_sums = torch.zeros_like(sums) if random_embedding is not None else None
    random_sums_sq = torch.zeros_like(sums) if random_embedding is not None else None
    count = 0
    batch: list[Sequence[int]] = []
    generator = torch.Generator(device=device).manual_seed(seed)

    def consume(items: list[Sequence[int]]) -> None:
        nonlocal count
        token_ids = torch.as_tensor(items, dtype=torch.long, device=device)
        x = embedding(token_ids)
        if centroid is not None:
            x = x - centroid
        coefficients = torch.einsum("lk,bld->bkd", basis, x)
        power = coefficients.float().square().sum(dim=-1)
        sums.add_(power.sum(dim=0).double().cpu())
        sums_sq.add_(power.double().square().sum(dim=0).cpu())
        if shuffled_sums is not None:
            # Independently permute positions in each sequence. This preserves
            # unigram content and removes sequential structure.
            order = torch.rand(x.shape[:2], device=device, generator=generator).argsort(
                1
            )
            shuffled = x.gather(1, order[..., None].expand_as(x))
            shuffled_coefficients = torch.einsum("lk,bld->bkd", basis, shuffled)
            shuffled_power = shuffled_coefficients.float().square().sum(dim=-1)
            shuffled_sums.add_(shuffled_power.sum(dim=0).double().cpu())
        if random_embedding is not None:
            random_x = random_embedding(token_ids)
            if random_centroid is not None:
                random_x = random_x - random_centroid
            random_coefficients = torch.einsum("lk,bld->bkd", basis, random_x)
            random_power = random_coefficients.float().square().sum(dim=-1)
            random_sums.add_(random_power.sum(dim=0).double().cpu())
            random_sums_sq.add_(random_power.double().square().sum(dim=0).cpu())
        count += len(items)

    for sequence in sequences:
        if len(sequence) != sequence_length:
            raise ValueError("all sequences must have exactly sequence_length tokens")
        batch.append(sequence)
        if len(batch) == batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    if count == 0:
        raise ValueError("the dataset produced no complete sequences")
    result: dict[str, torch.Tensor | int] = {
        "sum": sums,
        "sum_sq": sums_sq,
        "count": count,
    }
    if shuffled_sums is not None:
        result["shuffled_sum"] = shuffled_sums
    if random_sums is not None:
        result["random_sum"] = random_sums
        result["random_sum_sq"] = random_sums_sq
    return result


def summarize(stats: dict[str, torch.Tensor | int], embedding_dim: int):
    count = int(stats["count"])
    mean = stats["sum"] / count
    variance = (stats["sum_sq"] / count - mean.square()).clamp_min(0)
    stderr = (variance / count).sqrt()
    result = {
        "mean_power": mean.numpy(),
        "stderr_power": stderr.numpy(),
        "power_per_embedding_dim": (mean / embedding_dim).numpy(),
        "power_fraction": (mean / mean.sum()).numpy(),
        "cumulative_power_fraction": (mean.cumsum(0) / mean.sum()).numpy(),
    }
    if "shuffled_sum" in stats:
        shuffled = stats["shuffled_sum"] / count
        result["shuffled_mean_power"] = shuffled.numpy()
        result["power_over_shuffled"] = (mean / shuffled.clamp_min(1e-30)).numpy()
    if "random_sum" in stats:
        random_mean = stats["random_sum"] / count
        random_variance = (
            stats["random_sum_sq"] / count - random_mean.square()
        ).clamp_min(0)
        result["random_mean_power"] = random_mean.numpy()
        result["random_stderr_power"] = (random_variance / count).sqrt().numpy()
        result["learned_over_random_power"] = (
            mean / random_mean.clamp_min(1e-30)
        ).numpy()
    return result


def save_results(output: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output.with_suffix(".npz"), mode=np.arange(len(arrays["mean_power"])), **arrays
    )
    with output.with_suffix(".csv").open("w", newline="") as handle:
        names = list(arrays)
        writer = csv.writer(handle)
        writer.writerow(["mode", *names])
        for mode in range(len(arrays["mean_power"])):
            writer.writerow([mode, *(float(arrays[name][mode]) for name in names)])
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_plot(output.with_suffix(".svg"), arrays, metadata)


def save_plot(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    """Write a dependency-free SVG summary of the mode-power metrics."""
    width, height = 1200, 920
    left, right, panel_h = 95, 35, 205
    plot_w = width - left - right
    panels = [
        (85, "Mode power", True),
        (350, "Learned / random power", True),
        (615, "Cumulative power fraction", False),
    ]
    modes = np.arange(len(arrays["mean_power"]), dtype=float)
    colors = {
        "learned": "#276FBF",
        "random": "#E07A5F",
        "ratio": "#5B8E7D",
        "cumulative": "#6A4C93",
        "reference": "#777777",
    }

    def xmap(values: np.ndarray, log_x: bool) -> np.ndarray:
        transformed = np.log1p(values) if log_x else values
        maximum = math.log1p(modes[-1]) if log_x else modes[-1]
        return left + plot_w * transformed / maximum

    def polyline(
        x: np.ndarray,
        y: np.ndarray,
        color: str,
        y0: float,
        ymin: float,
        ymax: float,
        log_y: bool = False,
    ) -> str:
        if log_y:
            y, ymin, ymax = np.log10(y), math.log10(ymin), math.log10(ymax)
        py = y0 + panel_h * (1 - (y - ymin) / (ymax - ymin))
        points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, py, strict=True))
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:13px}"
        ".label{font-size:15px}.title{font-size:23px;font-weight:600}"
        ".panel{font-size:17px;font-weight:600}</style>",
        '<text class="title" x="600" y="35" text-anchor="middle">'
        "GPT-2 embedding power in the positional DCT basis</text>",
        f'<text class="label" x="600" y="59" text-anchor="middle">'
        f"{html.escape(str(metadata.get('num_sequences', '?')))} OWT sequences; "
        f"length {html.escape(str(metadata.get('sequence_length', len(modes))))}; "
        f"corpus-centered</text>",
    ]

    for y0, title, log_x in panels:
        svg.extend(
            [
                f'<text class="panel" x="{left}" y="{y0 - 16}">{title}</text>',
                f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{panel_h}" '
                'fill="none" stroke="#444"/>',
            ]
        )
        tick_values = (
            [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023]
            if log_x
            else [0, 256, 512, 768, 1023]
        )
        for tick, px in zip(
            tick_values, xmap(np.asarray(tick_values), log_x), strict=True
        ):
            svg.append(
                f'<line x1="{px:.2f}" y1="{y0 + panel_h}" x2="{px:.2f}" '
                f'y2="{y0 + panel_h + 5}" stroke="#444"/>'
            )
            svg.append(
                f'<text class="tick" x="{px:.2f}" y="{y0 + panel_h + 21}" '
                f'text-anchor="middle">{tick}</text>'
            )
        svg.append(
            f'<text class="label" x="600" y="{y0 + panel_h + 44}" '
            'text-anchor="middle">DCT mode k (low → high frequency)</text>'
        )

    # Panel 1 uses a logarithmic y scale so the random baseline remains visible.
    y0 = panels[0][0]
    learned = arrays["mean_power"]
    random = arrays.get("random_mean_power")
    plotted = learned if random is None else np.concatenate([learned, random])
    ymin, ymax = float(plotted.min()) * 0.8, float(plotted.max()) * 1.2
    svg.append(
        polyline(xmap(modes, True), learned, colors["learned"], y0, ymin, ymax, True)
    )
    if random is not None:
        svg.append(
            polyline(xmap(modes, True), random, colors["random"], y0, ymin, ymax, True)
        )
    for value in [0.3, 1, 3, 10, 30, 100]:
        if ymin <= value <= ymax:
            py = y0 + panel_h * (
                1
                - (math.log10(value) - math.log10(ymin))
                / (math.log10(ymax) - math.log10(ymin))
            )
            svg.append(
                f'<text class="tick" x="{left - 10}" y="{py + 4:.2f}" '
                f'text-anchor="end">{value:g}</text>'
            )
    svg.append(
        f'<text class="tick" x="{left + 14}" y="{y0 + 20}" fill="{colors["learned"]}">learned</text>'
    )
    if random is not None:
        svg.append(
            f'<text class="tick" x="{left + 80}" y="{y0 + 20}" fill="{colors["random"]}">random init</text>'
        )

    # Panel 2: learned/random effect size.
    y0 = panels[1][0]
    ratio = arrays.get("learned_over_random_power", np.ones_like(learned))
    rmin, rmax = 0.0, float(ratio.max()) * 1.1
    svg.append(polyline(xmap(modes, True), ratio, colors["ratio"], y0, rmin, rmax))
    for value in np.linspace(rmin, rmax, 5):
        py = y0 + panel_h * (1 - value / rmax)
        svg.append(
            f'<text class="tick" x="{left - 10}" y="{py + 4:.2f}" '
            f'text-anchor="end">{value:.0f}×</text>'
        )

    # Panel 3: cumulative learned power and the uniform-power reference.
    y0 = panels[2][0]
    cumulative = arrays["cumulative_power_fraction"]
    svg.append(polyline(xmap(modes, False), cumulative, colors["cumulative"], y0, 0, 1))
    svg.append(
        polyline(
            xmap(modes, False), (modes + 1) / len(modes), colors["reference"], y0, 0, 1
        )
    )
    for value in [0, 0.25, 0.5, 0.75, 1]:
        py = y0 + panel_h * (1 - value)
        svg.append(
            f'<text class="tick" x="{left - 10}" y="{py + 4:.2f}" '
            f'text-anchor="end">{value:.2f}</text>'
        )
    svg.append(
        f'<text class="tick" x="{left + 14}" y="{y0 + 20}" '
        f'fill="{colors["cumulative"]}">observed</text>'
    )
    svg.append(
        f'<text class="tick" x="{left + 90}" y="{y0 + 20}" '
        f'fill="{colors["reference"]}">uniform reference</text>'
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Explicit load_from_disk path; defaults to the repo's prepared OWT cache.",
    )
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/patrick/.cache/discrete_diffusion/owt"),
        help="Matches configs/data/openwebtext*.yaml via configs/config.yaml scratch_dir.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        help="Optional Hugging Face model cache; defaults to its standard location.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-corpus-centering",
        action="store_true",
        help="Do not subtract the empirical centroid of the analyzed token corpus.",
    )
    parser.add_argument(
        "--no-random-baseline",
        action="store_true",
        help="Disable the GPT-2-shaped Gaussian random embedding baseline.",
    )
    parser.add_argument("--shuffle-baseline", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/owt_gpt2_dct_power")
    )
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.sequence_length < 1 or args.num_sequences < 1 or args.batch_size < 1:
        raise ValueError(
            "sequence length, sequence count, and batch size must be positive"
        )
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(args.model_cache_dir) if args.model_cache_dir else None,
    )
    embedding = model.get_input_embeddings().to(device).eval()
    initializer_range = float(getattr(model.config, "initializer_range", 0.02))
    random_embedding = None
    if not args.no_random_baseline:
        # Construct on CPU with a private generator so this baseline is stable
        # across CPU/GPU selection and does not alter global RNG state.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        random_weight = torch.empty(
            embedding.num_embeddings,
            embedding.embedding_dim,
            dtype=embedding.weight.dtype,
        )
        random_weight.normal_(mean=0.0, std=initializer_range, generator=generator)
        random_embedding = (
            torch.nn.Embedding.from_pretrained(random_weight).to(device).eval()
        )
    del model

    configured_path = args.cache_dir / (
        f"openwebtext-train_train_bs{args.sequence_length}_wrapped.dat"
    )
    dataset_path = args.dataset_path or configured_path
    if dataset_path.exists():
        from datasets import load_from_disk

        dataset = load_from_disk(str(dataset_path))
        take = min(args.num_sequences, len(dataset))
        sequences = dataset.select(range(take))["input_ids"]
        data_source = str(dataset_path)
        packing = "repository-prepared wrapped dataset (dataloader.py)"
    else:
        print(
            f"Prepared dataset not found at {dataset_path}; falling back to "
            "streaming raw OWT."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=str(args.model_cache_dir) if args.model_cache_dir else None,
        )
        dataset = load_dataset(
            args.dataset,
            split=args.split,
            streaming=True,
            cache_dir=str(args.cache_dir),
        )
        texts = (example[args.text_column] for example in dataset)
        sequences = list(
            packed_token_sequences(
                texts, tokenizer, args.sequence_length, args.num_sequences
            )
        )
        data_source = f"huggingface:{args.dataset}:{args.split}"
        packing = "concatenated documents with EOS; incomplete tail dropped"
    centroid = None
    random_centroid = None
    if not args.no_corpus_centering:
        centroid = corpus_centroid(sequences, embedding)
        if random_embedding is not None:
            random_centroid = corpus_centroid(sequences, random_embedding)
    stats = accumulate_mode_power(
        sequences,
        embedding,
        args.sequence_length,
        args.batch_size,
        device,
        centroid=centroid,
        random_embedding=random_embedding,
        random_centroid=random_centroid,
        shuffle_baseline=args.shuffle_baseline,
        seed=args.seed,
    )
    arrays = summarize(stats, embedding.embedding_dim)
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "data_source": data_source,
        "split": args.split,
        "sequence_length": args.sequence_length,
        "num_sequences": int(stats["count"]),
        "embedding_dim": embedding.embedding_dim,
        "corpus_centered": not args.no_corpus_centering,
        "centroid_definition": (
            "token-weighted mean embedding over the exact packed sequences"
            if not args.no_corpus_centering
            else None
        ),
        "random_baseline": not args.no_random_baseline,
        "random_baseline_distribution": (
            f"Normal(0, {initializer_range}^2)" if not args.no_random_baseline else None
        ),
        "shuffle_baseline": args.shuffle_baseline,
        "seed": args.seed,
        "power_definition": "sum_d DCT(embedding)[mode,d]^2",
        "packing": packing,
    }
    save_results(args.output, arrays, metadata)
    print(
        f"Analyzed {stats['count']} sequences; wrote {args.output}.{{npz,csv,json,svg}}"
    )


if __name__ == "__main__":
    main()
