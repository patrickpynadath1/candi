#!/usr/bin/env python
"""Does self-conditioning change how the model should be sampled? (analysis only)

    python experiments/sc-sampler-choice/analyze.py RUN_ROOT [RUN_ROOT ...] [--output DIR]

Reads the sc-vs-depth runs (same roots as experiments/sc-vs-depth/analyze.py) and
prints, from validation scores only: the winning sampler per model, every policy's
score next to ``expected.csv``, and the SC / 8-layer / 16-layer comparison when every
model is restricted to one sampler. Writes ``DIR/sampler_choice.json``.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sc_vs_depth", HERE.parent / "sc-vs-depth" / "analyze.py")
sc_vs_depth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc_vs_depth)

POLICIES = sc_vs_depth.POLICIES
ARMS = ["none", "sc", "depth"]


def rows(cases):
    out = []
    for (task, method, arm), case in sorted(cases.items(), key=lambda item: sc_vs_depth_order(item[0])):
        winner, tied = case.winner()
        out.append(
            {
                "task": task,
                "method": method,
                "arm": arm,
                "winner": winner,
                "tie": tied if len(tied) > 1 else [],
                "validation": {name: round(p.validation, 4) for name, p in case.policies.items()},
            }
        )
    return out


def sc_vs_depth_order(key):
    names = [*sc_vs_depth.TASKS.values(), *sc_vs_depth.METHODS.values(), *ARMS]
    return [names.index(x) for x in key]


def winner_map(table):
    print("Winning sampler by validation (* = tie, first listed kept)")
    print(f"{'task':12s} {'method':9s} " + " ".join(f"{arm:16s}" for arm in ARMS))
    grid = {(r["task"], r["method"], r["arm"]): r for r in table}
    for task, method in dict.fromkeys((r["task"], r["method"]) for r in table):
        cells = []
        for arm in ARMS:
            r = grid.get((task, method, arm))
            cells.append(f"{(r['winner'] + ('*' if r['tie'] else '')) if r else '-':16s}")
        print(f"{task:12s} {method:9s} " + " ".join(cells))


def restricted(table, policy):
    """Validation of SC, 8 and 16 layers when every model uses ``policy``."""
    print(f"\nOnly {policy}: validation for 8L / 8L+SC / 16L (best of the three marked)")
    grid = {(r["task"], r["method"], r["arm"]): r["validation"].get(policy) for r in table}
    for task, method in dict.fromkeys((r["task"], r["method"]) for r in table):
        values = [grid.get((task, method, arm)) for arm in ARMS]
        present = [v for v in values if v is not None]
        best = max(present) if present else None
        text = [f"{'-' if v is None else f'{v:.4f}'}{'<' if v is not None and v == best else ' '}" for v in values]
        print(f"{task:12s} {method:9s} " + "  ".join(f"{t:>10s}" for t in text))


def compare_expected(table, path=HERE / "expected.csv"):
    grid = {(r["task"], r["method"], r["arm"]): r for r in table}
    mismatches = compared = 0
    print("\nComparison with expected.csv: differences (validation, winner) and the values the post quotes")
    for e in csv.DictReader(path.open()):
        r = grid.get((e["task"], e["method"], e["arm"]))
        if r is None or e["policy"] not in r["validation"]:
            continue
        compared += 1
        got, expected = r["validation"][e["policy"]], float(e["validation"])
        won = r["winner"] == e["policy"]
        if abs(got - expected) > 5e-5 or won != (e["winner"] == "yes"):
            mismatches += 1
            print(
                f"  {e['task']} {e['method']} {e['arm']} {e['policy']}: expected {expected:.4f}"
                f"{' (winner)' if e['winner'] else ''}, got {got:.4f}{' (winner)' if won else ''}"
            )
        if e["quoted"]:
            print(f"  quoted in the post: {e['task']} {e['method']} {e['arm']} {e['policy']} "
                  f"{got:.4f} [{e['quoted']}]")
    print(f"{mismatches} of {compared} expected policy scores differ")
    return mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    table = rows(sc_vs_depth.load(args.roots))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sampler_choice.json").write_text(json.dumps({"rows": table}, indent=1) + "\n")
    print(f"wrote {len(table)} rows to {args.output / 'sampler_choice.json'}\n")
    winner_map(table)
    restricted(table, "native")
    compare_expected(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
