"""Markdown tables from sudoku_hint_ladder.py's summary.json.

    python sudoku_ladder_report.py SUMMARY.json SPEC.json > report.md

Per recipe and sampler, at each retained-clue fraction: valid rate and distinct valid
boards. Below 0 hints all 32 puzzles are the same empty grid, so there the count is pooled
over all 512 generations; elsewhere it is the mean per puzzle (out of 16 generations).
"""

import json
import sys
from collections import defaultdict

FAMILY = {"masked": "masked", "uniform_posterior_ce": "uniform", "uniform": "uniform",
          "gaussian_embedding": "Gaussian", "simplex": "simplex", "spherical": "spherical",
          "spherical_vmf": "vMF"}


def label(case):
    body = case.removeprefix("sudoku_hard-").split("-seed-")[0]
    arm = "SC" if body.endswith("expected_embedding") else "no SC"
    parts = body.rsplit("-", 1)[0].split("-")
    name = " + ".join(FAMILY[p] for p in parts)
    return name, arm


def main(summary_path, spec_path):
    rows = json.load(open(summary_path))
    spec = {c["case"]: c for c in json.load(open(spec_path))}
    fracs = sorted({r["fraction"] for r in rows}, reverse=True)
    by = defaultdict(dict)
    for r in rows:
        by[(r["case"], r["sampler"])][r["fraction"]] = r
    hints = {f: round(sum(r["mean_hints"] for r in rows if r["fraction"] == f) /
                      sum(1 for r in rows if r["fraction"] == f), 1) for f in fracs}
    head = " | ".join(f"{hints[f]:g} hints" for f in fracs)
    cell = lambda r, f: "–" if r is None else (
        f"{r['valid_rate']:.2f} / {r['pooled_valid_unique'] if f == 0 else round(r['valid_unique_per_puzzle'], 1)}")
    print("Cells: valid rate / distinct valid boards (per puzzle of 16; at 0 hints pooled of 512).\n")
    print(f"| recipe | SC | sampler | val acc | {head} |")
    print("|" + "---|" * (4 + len(fracs)))
    order = sorted(spec, key=lambda c: (label(c)[0].count("+"), label(c)))
    for case in order:
        name, arm = label(case)
        for s in sorted(spec[case]["samplers"], key=lambda s: (s["rank"] == 0, s["rank"])):
            got = by.get((case, s["name"]), {})
            val = "untuned" if s["val"] is None else f"{s['val']:.2f}"
            cells = " | ".join(cell(got.get(f), f) for f in fracs)
            print(f"| {name} | {arm} | {s['name']} | {val} | {cells} |")
    # one line per recipe: best over its 4 samplers at 0 hints, by distinct valid boards
    print("\nBest sampler per recipe at 0 hints (most distinct valid boards of 512):\n")
    print("| recipe | SC | sampler | valid rate | distinct valid |")
    print("|---|---|---|---|---|")
    best = []
    for case in order:
        cand = [(by[(case, s["name"])].get(0.0), s["name"]) for s in spec[case]["samplers"]]
        cand = [(r, n) for r, n in cand if r]
        if cand:
            r, n = max(cand, key=lambda x: (x[0]["pooled_valid_unique"], x[0]["valid_rate"]))
            best.append((r["pooled_valid_unique"], case, n, r))
    for v, case, n, r in sorted(best, reverse=True):
        name, arm = label(case)
        print(f"| {name} | {arm} | {n} | {r['valid_rate']:.2f} | {v} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
