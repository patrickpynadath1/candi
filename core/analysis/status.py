"""Where each case of a nested-tuning run stands, read from its output tree.

    python -m core.analysis.status RUN_ROOT [--json]

States, first match wins: ``done`` (final/result.json), ``running`` (a file
under the case changed in the last ``--active-minutes``), ``failed`` (a failed or
skipped candidate and nothing running), ``partial`` (some candidates or policies
selected) and ``not started``. Reads files and their mtimes only: it never takes
the driver lock, so it cannot disturb a live run.
"""

import argparse
import json
import time
from pathlib import Path


def _idle_minutes(path):
    """Minutes since any file under ``path`` last changed."""
    newest = max((p.stat().st_mtime for p in path.rglob("*") if p.is_file()), default=0)
    return (time.time() - newest) / 60


def _lines(path):
    return [line for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def case_status(path, active_minutes=30):
    """One case directory as a dict: name, state, and how far it got."""
    path = Path(path)
    spec = json.loads((path / "protocol.json").read_text()).get("spec", {})
    policies = [policy["name"] for policy in spec.get("inference", [])]
    candidates = sorted(c for c in (path / "training" / "candidates").glob("*") if c.is_dir())
    trained = [c for c in candidates if (c / "result.json").exists()]
    failed = [c.name for c in candidates if (c / "failure.json").exists()]
    skipped = len(_lines(path / "training" / "skipped.jsonl")) + sum(
        len(_lines(p)) for p in path.glob("training/candidates/*/inference/*/skipped.jsonl")
    )
    selected = sorted({p.parent.name for p in path.glob("training/candidates/*/inference/*/selected.json")})
    idle = _idle_minutes(path)
    if (path / "final" / "result.json").exists():
        state = "done"
    elif idle < active_minutes:
        state = "running"
    elif failed or skipped:
        state = "failed"
    elif candidates or selected:
        state = "partial"
    else:
        state = "not started"
    return {
        "case": spec.get("name", path.name),
        "state": state,
        "trained": f"{len(trained)}/{spec.get('training', {}).get('trials', '?')}",
        "policies": f"{len(selected)}/{len(policies)}",
        "failed": failed,
        "skipped": skipped,
        "idle_minutes": round(idle, 1),
        "path": str(path),
    }


def run_status(root, active_minutes=30):
    return [case_status(p.parent, active_minutes) for p in sorted(Path(root).rglob("protocol.json"))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="nested_tuning --output directory")
    parser.add_argument("--json", action="store_true", help="print one JSON list instead of a table")
    parser.add_argument("--active-minutes", type=float, default=30,
                        help="a case with a file newer than this counts as running (default 30)")
    args = parser.parse_args(argv)
    rows = run_status(args.root, args.active_minutes)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print(f"no cases under {args.root} (no protocol.json found)")
        return
    width = max(len(row["case"]) for row in rows)
    print(f"{'case':{width}}  {'state':11}  trained  policies  skipped  idle (min)")
    for row in rows:
        print(f"{row['case']:{width}}  {row['state']:11}  {row['trained']:>7}  {row['policies']:>8}  "
              f"{row['skipped']:>7}  {row['idle_minutes']:>10}")
    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print(", ".join(f"{n} {state}" for state, n in sorted(counts.items())))


if __name__ == "__main__":
    main()
