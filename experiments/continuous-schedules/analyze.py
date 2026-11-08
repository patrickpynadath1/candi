"""Tables for the continuous-schedules example (the post's cdcd_*.json).

python experiments/continuous-schedules/analyze.py RUN_ROOT [RUN_ROOT ...] [--output DIR]

Reads nested-tuning case directories under each RUN_ROOT and writes:

- ``cdcd_performance.json``: one row per (task, method, SC arm) with the test
  score of the validation-selected sampler, per-seed scores, validation score,
  every sampler's validation result, and BLEU without the brevity penalty for
  TinyStories. A case tuned in several runs (one checkpoint, more sampler
  policies) is merged; the sampler is chosen by validation over all of them.
- ``cdcd_schedules.json``: when the run tree still holds checkpoints, the
  learned schedules, loss bins (validation and training) and corruption
  curves of every case, in the schema of the post's data file.

It then compares both with ``expected.csv`` (the published numbers).
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the repository root, for core
sys.path.insert(0, str(HERE))

from core.analysis import bleu_parts, load_cases, merge_cases  # noqa: E402

TASKS = {"sudoku_hard": "sudoku_hard", "text8": "text8", "tinystories-bpe-10000": "tinystories"}
METHODS = {"gaussian_embedding": "gaussian", "simplex": "simplex", "spherical": "spherical", "spherical_vmf": "vmf"}
ARMS = {"none": "none", "expected_embedding": "sc"}
# The MI210 TinyStories runs named the renoising samplers differently.
POLICY_NAMES = {"renoising": "renoise_sample", "renoising_greedy": "renoise_greedy"}
ORDER = [
    (task, method, arm)
    for method in METHODS.values()
    for task in ("sudoku_hard", "text8")
    for arm in ARMS.values()
] + [("tinystories", method, arm) for method in METHODS.values() for arm in ARMS.values()]


def cell(case):
    """(task, method, arm) of a case, or None if it is not part of this example."""
    path, _, feedback = case.recipe.rpartition("-")
    if case.task not in TASKS or path not in METHODS or feedback not in ARMS:
        return None
    return TASKS[case.task], METHODS[path], ARMS[feedback]


def policy_name(name):
    return POLICY_NAMES.get(name, name)


def short(parameters):
    """Sampler parameters by their last key segment, floats to 3 decimals."""
    return {
        key.rsplit(".", 1)[-1]: round(value, 3) if isinstance(value, float) else value
        for key, value in parameters.items()
    }


def r4(values):
    return [round(v, 4) for v in values] if isinstance(values, list) else round(values, 4)


def performance_row(case):
    task, method, arm = cell(case)
    winner, tied = case.winner()
    row = {"task": task, "method": method, "arm": arm}
    if winner is None or case.test is None:
        return {**row, "pending": True, "case": case.name}
    if policy_name(case.selected) != policy_name(winner):
        raise ValueError(f"{case.name}: test score is from {case.selected}, validation picks {winner}")
    row.update(
        score=r4(case.test),
        seeds=r4(case.test_seeds),
        validation=r4(case.policies[winner].validation),
        policy=policy_name(winner),
        renoise_tried=any(policy_name(name).startswith("renoise") for name in case.policies),
        case=case.name,
        samplers=[
            {
                "policy": policy_name(policy.name),
                "validation": r4(policy.validation),
                "seeds": r4(policy.seeds),
                "params": short(policy.parameters),
            }
            for policy in case.policies.values()
        ],
    )
    if len(tied) > 1:
        row["tie"] = [policy_name(name) for name in tied]
    if task == "tinystories":
        parts = [bleu_parts(records) for records in case.records().values()]
        if parts:
            row["score_nobp"] = r4(sum(p["bleu_nobp"] for p in parts) / len(parts))
            row["seeds_nobp"] = [r4(p["bleu_nobp"]) for p in parts]
            row["bp"] = r4(sum(p["bp"] for p in parts) / len(parts))
            row["len_ratio"] = r4(sum(p["len_ratio"] for p in parts) / len(parts))
    return row


def trained_records(case):
    """trained.json records of the case's training candidates whose checkpoint exists."""
    records = []
    for path in sorted(case.path.glob("training/candidates/*/trained.json")):
        record = json.loads(path.read_text())
        if Path(record.get("checkpoint", "")).is_file() and Path(record.get("config", "")).is_file():
            records.append(record)
    return records


def schedule_tables(cases, args):
    """cdcd_schedules.json content for every case with a checkpoint, or None."""
    import corruption_report
    import schedule_report

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"source": "experiments/continuous-schedules/analyze.py (schedule_report.py, corruption_report.py)",
           "cases": [], "errors": [], "schedules": {}, "bins": {}, "bins_train": {}, "corruption": {},
           "corruption_meta": {"source": "experiments/continuous-schedules/corruption_report.py",
                               "samples_per_point": args.samples, "cases": [],
                               "decoder": corruption_report.DECODER, "rank": corruption_report.RANK}}
    for key, case in cases:
        records = trained_records(case)
        if not records:
            continue
        if len(records) > 1:
            out["errors"].append({"case": case.name, "error": f"{len(records)} trained checkpoints; expected one"})
            continue
        key = "|".join(key)
        try:
            tables = schedule_report.schedule_case(
                records[0], bins=args.bins, examples=args.examples, grid=args.grid, device=device
            )
            corruption = corruption_report.corruption_case(
                records[0], samples=args.samples, points=args.points, device=device
            )
        except Exception as error:  # one broken case must not hide the others
            out["errors"].append({"case": case.name, "error": repr(error)})
            print(f"ERROR {case.name}: {error!r}", file=sys.stderr)
            continue
        out["cases"].append({"case": case.name, **tables["meta"]})
        out["schedules"][key] = tables["schedule"]
        out["bins"][key] = tables["bins"]
        out["bins_train"][key] = tables["bins_train"]
        out["corruption"][key] = corruption["corruption"]
        out["corruption_meta"]["cases"].append({"case": case.name, **corruption["meta"]})
    return out if out["cases"] or out["errors"] else None


def read_expected(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def compare(expected, rows, schedules):
    """Print measured next to published values; return the number of compared cells."""
    by_cell = {(r["task"], r["method"], r["arm"]): r for r in rows}
    compared = 0
    print(f"{'quantity':26} {'task':12} {'method':10} {'arm':5} {'published':>10} {'measured':>10} {'delta':>9}  policy (published/measured)")
    for entry in expected:
        key = (entry["task"], entry["method"], entry["arm"])
        published = float(entry["value"])
        measured, policies = None, ""
        if entry["quantity"] == "score":
            row = by_cell.get(key)
            if row and not row.get("pending"):
                measured = row["score"]
                policies = f"{entry['policy']}/{row['policy']}"
        elif schedules is not None:
            from schedule_report import linearity

            split = "bins" if entry["quantity"].endswith("validation") else "bins_train"
            bins = schedules[split].get("|".join(key))
            if bins:
                measured = round(linearity(bins), 5)
        if measured is None or (isinstance(measured, float) and math.isnan(measured)):
            continue
        compared += 1
        print(f"{entry['quantity']:26} {key[0]:12} {key[1]:10} {key[2]:5} {published:10.4f} {measured:10.4f} "
              f"{measured - published:+9.4f}  {policies}")
    return compared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="nested-tuning output directories")
    parser.add_argument("--output", type=Path, default=Path("."), help="where to write the JSON files")
    parser.add_argument("--expected", type=Path, default=HERE / "expected.csv")
    parser.add_argument("--no-schedules", action="store_true", help="skip the checkpoint analyses")
    parser.add_argument("--device", default="auto", help="cpu, cuda or auto")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--points", type=int, default=101)
    args = parser.parse_args(argv)

    cases = merge_cases([case for root in args.roots for case in load_cases(root) if cell(case)])
    by_cell = {}
    for case in cases:
        if cell(case) in by_cell:
            raise ValueError(f"two cases for {cell(case)}: {by_cell[cell(case)].name}, {case.name}")
        by_cell[cell(case)] = case
    ordered = [(key, by_cell[key]) for key in ORDER if key in by_cell]
    rows = [performance_row(case) for _, case in ordered]
    args.output.mkdir(parents=True, exist_ok=True)
    performance = {"source": "experiments/continuous-schedules/analyze.py over " + ", ".join(
        root.name for root in args.roots) + "; sampler chosen by validation", "rows": rows}
    (args.output / "cdcd_performance.json").write_text(json.dumps(performance, indent=1) + "\n")
    print(f"wrote {args.output / 'cdcd_performance.json'}: {len(rows)} rows")

    schedules = None if args.no_schedules else schedule_tables(ordered, args)
    if schedules is not None:
        (args.output / "cdcd_schedules.json").write_text(json.dumps(schedules) + "\n")
        print(f"wrote {args.output / 'cdcd_schedules.json'}: {len(schedules['cases'])} cases, "
              f"{len(schedules['errors'])} errors")
    elif not args.no_schedules:
        print("no checkpoints under the run roots; skipped the schedule tables")
    if args.expected.exists():
        print(f"\ncompared {compare(read_expected(args.expected), rows, schedules)} values with {args.expected}")
    return performance, schedules


if __name__ == "__main__":
    main()
