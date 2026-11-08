"""Tokenizer-ladder table: run trees -> vocab_ladder.json (the website's schema).

    python experiments/tokenizer-ladder/analyze.py RUN_ROOT [RUN_ROOT ...] [--output DIR]

Reads every nested-tuning case under the roots (``core.analysis``), keeps the
TinyStories BPE cases of the ladder's seven recipes, and writes one row per
(vocabulary, recipe): the validation score of every sampler policy, the policy
chosen on validation, and its test BLEU per seed. Test BLEU is recomputed from the
stored test records with and without the brevity penalty (``bleu_parts``), with the
brevity penalty and the length ratio. Then the rows are compared with expected.csv.
Needs the ``scoring`` extra (sacrebleu).
"""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.analysis import bleu_parts, load_cases, merge_cases  # noqa: E402

HERE = Path(__file__).resolve().parent
# Recipe name (benchmark.algorithm_name) -> (method, self-conditioning).
RECIPES = {
    "ar-8": ("ar", None),
    "gaussian_embedding-none": ("gaussian", False),
    "gaussian_embedding-expected_embedding": ("gaussian", True),
    "masked-none": ("masked", False),
    "masked-expected_embedding": ("masked", True),
    "uniform_posterior_ce-none": ("uniform", False),
    "uniform_posterior_ce-expected_embedding": ("uniform", True),
}
METHODS = ["ar", "gaussian", "masked", "uniform"]
# Policy names in the runs -> names on the website.
POLICIES = {"renoising": "renoise_sample", "renoising_greedy": "renoise_greedy"}


def mean(values):
    return sum(values) / len(values)


def short(parameters):
    """{'algorithm.predictor.temperature': 0.8277} -> {'temperature': 0.828}."""
    return {
        key.rsplit(".", 1)[-1]: round(value, 3) if isinstance(value, float) else value
        for key, value in parameters.items()
    }


def row(case):
    method, sc = RECIPES[case.recipe]
    winner, tied = case.winner()
    if case.selected is not None and case.selected != winner:
        print(f"warning: {case.name} tested {case.selected}, validation winner {winner}")
    if len(tied) > 1:
        print(f"note: {case.name} validation tie {tied}; first listed wins")
    result = {
        "vocab": int(case.task.rsplit("-", 1)[1]),
        "method": method,
        "sc": sc,
        "chosen": POLICIES.get(case.selected, case.selected),
        "test": None if case.test is None else round(case.test, 3),
        "test_seeds": [round(score, 3) for score in case.test_seeds],
        "samplers": {
            POLICIES.get(name, name): {
                "val": round(policy.validation, 3),
                "val_seeds": [round(score, 3) for score in policy.seeds],
                "params": short(policy.parameters),
            }
            for name, policy in case.policies.items()
        },
    }
    records = case.records()
    if records:
        parts = [bleu_parts(records[seed]) for seed in sorted(records)]
        result.update(
            test_nobp=round(mean([p["bleu_nobp"] for p in parts]), 4),
            test_seeds_nobp=[round(p["bleu_nobp"], 4) for p in parts],
            test_bp=round(mean([p["bp"] for p in parts]), 4),
            test_len_ratio=round(mean([p["len_ratio"] for p in parts]), 4),
        )
    return result


def source(cases):
    """One line describing the evaluation protocol, read from the first case."""
    evaluation = json.loads((cases[0].path / "protocol.json").read_text())["spec"]["evaluation"]
    return (
        "TinyStories BPE, corpus BLEU; sampler scores are validation "
        f"({len(evaluation['tuning_indices'])} examples, seeds "
        f"{'/'.join(map(str, evaluation['seeds']))}), test is "
        f"{len(evaluation['test_indices'])} examples, seeds "
        f"{'/'.join(map(str, evaluation['test_seeds']))}, winner only"
    )


def table(roots):
    cases = merge_cases([case for root in roots for case in load_cases(root)])
    cases = [c for c in cases if c.task.startswith("tinystories-bpe-") and c.recipe in RECIPES]
    if not cases:
        raise SystemExit(f"No tokenizer-ladder cases under {', '.join(map(str, roots))}")
    rows = sorted(
        (row(case) for case in cases),
        key=lambda r: (METHODS.index(r["method"]), bool(r["sc"]), r["vocab"]),
    )
    return {"source": source(cases), "rows": rows}


def key(vocab, method, sc):
    return (int(vocab), method, None if sc in (None, "", "None") else str(sc) == "True")


def compare(rows, expected_path=HERE / "expected.csv"):
    """Print each row next to the published numbers; return the number of mismatches."""
    with open(expected_path) as handle:
        expected = {key(r["vocab"], r["method"], r["sc"]): r for r in csv.DictReader(handle)}
    fields = ["test", "test_nobp", "test_bp", "test_len_ratio"]
    print(f"{'vocab':>6} {'method':9} {'sc':5} {'chosen':15} " + " ".join(f"{f:>15}" for f in fields))
    mismatches = 0
    for r in rows:
        published = expected.get(key(r["vocab"], r["method"], r["sc"]))
        cells = []
        for field in fields:
            value = r.get(field)
            if published is None or value is None:
                cells.append(f"{value!s:>15}")
                continue
            delta = value - float(published[field])
            cells.append(f"{value:>8.4g} ({delta:+.3f})")
            mismatches += abs(delta) > 5e-4
        if published is not None and published["chosen"] != r["chosen"]:
            mismatches += 1
            cells.append(f"chosen differs: published {published['chosen']}")
        print(f"{r['vocab']:>6} {r['method']:9} {r['sc']!s:5} {r['chosen']!s:15} " + " ".join(cells))
    missing = set(expected) - {key(r["vocab"], r["method"], r["sc"]) for r in rows}
    print(f"{len(rows)} rows, {mismatches} differences from expected.csv, {len(missing)} expected rows missing")
    return mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="nested-tuning output directories")
    parser.add_argument("--output", type=Path, default=Path("."), help="directory for vocab_ladder.json")
    args = parser.parse_args(argv)
    data = table(args.roots)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "vocab_ladder.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}")
    compare(data["rows"])
    return data


if __name__ == "__main__":
    main()
