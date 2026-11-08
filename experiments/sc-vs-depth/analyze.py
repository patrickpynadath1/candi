#!/usr/bin/env python
"""Self-conditioning vs depth: nested-tuning run trees -> the post's table.

    python experiments/sc-vs-depth/analyze.py RUN_ROOT [RUN_ROOT ...] [--output DIR]
        [--website sc_depth_results.json]

Reads every case under the given roots (several roots are merged by case name, e.g.
a training run and a later run that tuned more samplers on the same checkpoints),
writes ``DIR/sc_depth_results.json`` in the website's schema and prints it next to
``expected.csv``. ``--website`` also diffs every field against a published JSON.
TinyStories BLEU without the brevity penalty needs the ``scoring`` extra (sacrebleu).
"""

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from core.analysis import POLICY_ALIASES as ALIASES, bleu_summary, load_cases, merge_cases  # noqa: E402

TASKS = {"sudoku_hard": "sudoku_hard", "text8": "text8", "tinystories-bpe-10000": "tinystories"}
METHODS = {"gaussian_embedding": "gaussian", "masked": "masked", "uniform_posterior_ce": "uniform"}
ARMS = {"none": "none", "expected_embedding": "sc", "double_depth": "depth"}
POLICIES = ["native", "native_greedy", "eb_greedy", "renoise_sample", "renoise_greedy"]


def cell(case):
    """(task, method, arm) of a grid case, or None for any other case."""
    path, _, feedback = case.recipe.partition("-")
    if case.task in TASKS and path in METHODS and feedback in ARMS:
        return TASKS[case.task], METHODS[path], ARMS[feedback]
    return None


def order(policies):
    names = POLICIES + sorted(set(policies) - set(POLICIES))
    return {name: policies[name] for name in names if name in policies}


def rename(case):
    policies = {ALIASES.get(name, name): policy for name, policy in case.policies.items()}
    for name, policy in policies.items():
        policy.name = name
    case.policies = order(policies)
    case.selected = ALIASES.get(case.selected, case.selected)
    return case


def first_policy(case):
    return min((POLICIES.index(n) for n in case.policies if n in POLICIES), default=len(POLICIES))


def load(roots):
    """{(task, method, arm): merged Case} over every grid case under ``roots``.

    Validation ties go to the first policy in POLICIES order, whatever the order of
    the roots: runs holding earlier policies are merged first.
    """
    cases = [rename(case) for root in roots for case in load_cases(root) if cell(case)]
    merged = merge_cases(sorted(cases, key=first_policy))
    for case in merged:
        case.policies = order(case.policies)
    return {cell(case): case for case in merged}


def r(value, digits=4):
    return round(value, digits) if isinstance(value, float) else value


def row(key, case):
    task, method, arm = key
    winner, tied = case.winner()
    if case.selected != winner:
        print(f"warning: {case.name}: test score is from {case.selected}, validation winner is {winner}")
    final = case.final_path.parent if case.final_path else case.path
    out = {
        "task": task,
        "method": method,
        "arm": arm,
        "score": r(case.test),
        "validation": r(case.policies[winner].validation) if winner else None,
        "seeds": [r(s) for s in case.test_seeds],
        "policy": case.selected,
        "campaign": final.parent.name,
        "case": case.name,
    }
    if len(tied) > 1:
        out["tie"] = tied
    out["renoise_tried"] = any(name.startswith("renoise") for name in case.policies)
    if task == "tinystories" and case.final_path is not None:
        out.update(bleu_summary(records for _, records in sorted(case.records().items())))
    out["samplers"] = [
        {
            "policy": name,
            "validation": r(policy.validation),
            "seeds": [r(s) for s in policy.seeds],
            "params": {k.rsplit(".", 1)[-1]: r(v, 3) for k, v in policy.parameters.items()},
        }
        for name, policy in case.policies.items()
    ]
    return out


def table(cases):
    order = {name: i for i, name in enumerate([*TASKS.values(), *METHODS.values(), *ARMS.values()])}
    return [row(key, cases[key]) for key in sorted(cases, key=lambda k: [order[x] for x in k])]


def compare_expected(rows, path=HERE / "expected.csv"):
    got = {(x["task"], x["method"], x["arm"]): x for x in rows}
    print(f"{'task':12s} {'method':9s} {'arm':6s} {'expected':>9s} {'got':>9s} {'delta':>8s}  policy (expected -> got)")
    for e in csv.DictReader(path.open()):
        x = got.get((e["task"], e["method"], e["arm"]))
        expected = float(e["score"])
        if x is None or x["score"] is None:
            print(f"{e['task']:12s} {e['method']:9s} {e['arm']:6s} {expected:9.4f} {'-':>9s}")
            continue
        flag = "" if x["policy"] == e["policy"] else "  *"
        print(
            f"{e['task']:12s} {e['method']:9s} {e['arm']:6s} {expected:9.4f} {x['score']:9.4f} "
            f"{x['score'] - expected:+8.4f}  {e['policy']} -> {x['policy']}{flag}"
        )


def compare_website(rows, path):
    """Field-by-field differences from a published sc_depth_results.json."""
    published = {(x["task"], x["method"], x["arm"]): x for x in json.loads(Path(path).read_text())["rows"]}
    got = {(x["task"], x["method"], x["arm"]): x for x in rows}
    differences = 0
    for key in sorted(set(published) | set(got)):
        a, b = published.get(key), got.get(key)
        if a is None or b is None:
            print(f"{key}: only in {'published' if b is None else 'this run'}")
            differences += 1
            continue
        for field in sorted(set(a) | set(b)):
            if field not in a:
                continue  # extra fields (renoise_tried on every row) are additions
            if a.get(field) != b.get(field):
                # campaign is the run directory's name, a label rather than a result.
                label = field == "campaign"
                print(f"{key} {field}{' (label)' if label else ''}: published={a.get(field)!r} got={b.get(field)!r}")
                differences += not label
    print(f"{differences} differences in results from {path}")
    return differences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--website", type=Path, help="published sc_depth_results.json to diff against")
    args = parser.parse_args(argv)
    rows = table(load(args.roots))
    args.output.mkdir(parents=True, exist_ok=True)
    data = {"source": "experiments/sc-vs-depth/analyze.py; per cell, the sampler with the best validation score", "rows": rows}
    (args.output / "sc_depth_results.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(rows)} rows to {args.output / 'sc_depth_results.json'}\n")
    compare_expected(rows)
    if args.website:
        print()
        return 1 if compare_website(rows, args.website) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
