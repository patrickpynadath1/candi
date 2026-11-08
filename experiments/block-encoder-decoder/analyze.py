"""Block diffusion: scale the encoder or the decoder? Tables from nested-tuning runs.

    python experiments/block-encoder-decoder/analyze.py RUN_ROOT [RUN_ROOT ...] [--output DIR]

Writes DIR/block_scaling.json in the website's schema (one row per block model and
task) and prints each cell next to the published value in expected.csv.

Per row: ``score`` and ``seeds`` are the selected policy's test score and per-seed
scores; ``validation`` is that policy's validation score; ``nfe`` is the mean NFE of
the test generations. Timing comes from the run tree, as for the published numbers:

- ``train_hours``: mtime of the candidate's ``trained.json`` minus the mtime of the
  case's ``protocol.json``, in hours. This is wall time from the case start to the
  recorded checkpoint, so it includes data preparation, compilation and the training
  callback. ``protocol.json`` is rewritten when a case resumes; if it is newer than
  ``trained.json`` the candidate's ``training/config.yaml`` is the start instead.
- ``eval_seconds``: mean over the test seeds of ``evaluation_seconds`` in
  ``final/seed-*/results.json`` (generating the 256 test examples once).

TinyStories rows also decompose corpus BLEU: ``bp`` (brevity penalty), ``len_ratio``
(generated / reference length) and ``score_nobp`` / ``seeds_nobp`` (BLEU / BP), from
the stored test generations. That needs sacrebleu (the ``scoring`` extra).
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.analysis import bleu_parts, load_cases, merge_cases  # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = {"sudoku_hard": "sudoku_hard", "text8": "text8", "tinystories-bpe-10000": "tinystories"}
# Row order of the published figure: encoder arm, decoder arm, then the controls.
ORDER = [
    ("e2d2", 1, 8), ("e2d2", 2, 8), ("e2d2", 4, 8), ("e2d2", 8, 1), ("e2d2", 8, 2),
    ("e2d2", 8, 4), ("e2d2", 8, 8), ("raw", 0, 8), ("shared", None, 8),
]


def architecture(config):
    """(kind, encoder layers, decoder layers) from a case's recorded config; None if not block diffusion."""
    config = OmegaConf.create(config)
    if not OmegaConf.select(config, "block_diffusion.enabled", default=False):
        return None
    decoder = OmegaConf.select(config, "model.n_blocks")
    if OmegaConf.select(config, "encoder_decoder") is None:
        return "shared", None, decoder
    if OmegaConf.select(config, "encoder_decoder.context_attention") is not None:
        return None  # encoder K/V depth mixtures: not part of this example
    encoder = OmegaConf.select(config, "encoder_decoder.encoder_layers")
    return ("raw" if encoder == 0 else "e2d2"), encoder, decoder


def train_hours(case):
    trained = sorted(case.path.glob("training/candidates/*/trained.json"))
    if not trained:
        return None
    end = trained[0].stat().st_mtime
    start = (case.path / "protocol.json").stat().st_mtime
    if start > end:
        start = (trained[0].parent / "training/config.yaml").stat().st_mtime
    return round((end - start) / 3600, 2)


def row(case):
    kind, encoder, decoder = architecture(case.config)
    out = dict(task=TASKS.get(case.task, case.task), kind=kind, enc=encoder, dec=decoder,
               name=case.name, status="completed" if case.final_path else "incomplete")
    if case.final_path is None:
        return out
    seeds = [json.loads(path.read_text()) for path in sorted(case.final_path.glob("seed-*/results.json"))]
    out.update(
        score=case.test,
        validation=case.policies[case.selected].validation if case.selected in case.policies else None,
        seeds=case.test_seeds,
        policy=case.selected,
        train_hours=train_hours(case),
        eval_seconds=statistics.mean(s["evaluation_seconds"] for s in seeds),
        nfe=statistics.mean(s["mean_nfe"] for s in seeds),
    )
    if out["task"] == "tinystories":
        parts = [bleu_parts(s["records"]) for s in seeds]
        out.update(
            score_nobp=round(statistics.mean(p["bleu_nobp"] for p in parts), 4),
            seeds_nobp=[round(p["bleu_nobp"], 4) for p in parts],
            bp=round(statistics.mean(p["bp"] for p in parts), 4),
            len_ratio=round(statistics.mean(p["len_ratio"] for p in parts), 4),
        )
    return out


def key(row):
    return row["task"], row["kind"], row["enc"], row["dec"]


def expected():
    with open(HERE / "expected.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["enc"] = int(row["enc"]) if row["enc"] else None
        row["dec"] = int(row["dec"])
    return {key(row): row for row in rows}


def sort_key(row):
    tasks = list(TASKS.values())
    task = tasks.index(row["task"]) if row["task"] in tasks else len(tasks)
    shape = (row["kind"], row["enc"], row["dec"])
    return task, ORDER.index(shape) if shape in ORDER else len(ORDER), row["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="nested-tuning output directories")
    parser.add_argument("--output", type=Path, default=Path("."), help="where block_scaling.json goes")
    args = parser.parse_args()
    cases = merge_cases([case for root in args.roots for case in load_cases(root)])
    cases = [case for case in cases if case.task in TASKS and architecture(case.config)]
    if not cases:
        raise SystemExit(f"No block-encoder-decoder cases under {args.roots}")
    rows = sorted((row(case) for case in cases), key=sort_key)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = dict(
        source=", ".join(str(root) for root in args.roots),
        note="final test scores, 2 eval seeds, one training seed; block models tuned with native and native_greedy only",
        rows=rows,
    )
    path = args.output / "block_scaling.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")

    published = expected()
    print(f"{'task':12} {'model':>11} {'policy':>13} {'score':>8} {'published':>9} {'train h':>7} {'eval s':>6}")
    for r in rows:
        shape = "shared-8" if r["kind"] == "shared" else f"{r['enc']}/{r['dec']}"
        reference = published.get(key(r), {}).get("score")
        reference = f"{float(reference):9.4f}" if reference else f"{'-':>9}"
        if r["status"] != "completed":
            print(f"{r['task']:12} {shape:>11} {'(incomplete)':>13} {'-':>8} {reference}")
            continue
        hours = f"{r['train_hours']:7.2f}" if r["train_hours"] is not None else f"{'-':>7}"
        print(f"{r['task']:12} {shape:>11} {r['policy']:>13} {r['score']:8.4f} {reference} {hours} {r['eval_seconds']:6.1f}")
    print(f"Wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
