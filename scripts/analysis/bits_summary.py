"""Join bits_recovered.py outputs with tuned test scores and rank-correlate within each task.

usage: python bits_summary.py BITS_DIR SC_RERUN_JSON > summary.md
Never compares across tasks. Features per model (SC models use their second pass):
  b_low   model bits at the lowest target capacity (2% of H): solving from context alone
  code    mean over capacities of (model bits - channel bits): bits taken from other positions
  gap     mean over capacities of (H - model bits): missing bits, flat weighting
  lik     mean over capacities of (H - model bits) / (H - channel bits): missing share,
          likelihood weighting (exact NELBO / H for masked diffusion)
"""

import json
import sys
from pathlib import Path


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12


def features(result):
    key = "bits_sc2" if result["self_conditioning"] else "bits_model"
    rows, h = result["rows"], result["entropy_bits"]
    model = [r[key] for r in rows]
    # likelihood weighting: of the bits the channel leaves open at c, the share the model
    # misses; H * its mean over c is the masked-diffusion NELBO in bits per token.
    lik = sum((h - m) / (h - r["bits_channel"]) for m, r in zip(model, rows)) / len(rows)
    return dict(
        lik=lik,
        b_low=model[0],
        code=sum(m - r["bits_channel"] for m, r in zip(model, rows)) / len(rows),
        gap=sum(h - m for m in model) / len(rows),
    )


def main(bits_dir, scores_path):
    data = json.loads(Path(scores_path).read_text())
    cases = {c["id"]: c for c in data["cases"] + data.get("duplicates", [])}
    table = []
    for path in sorted(Path(bits_dir).glob("*__*.json")):
        result = json.loads(path.read_text())
        case_id = path.stem.replace("__", "/")
        if case_id not in cases:
            print(f"<!-- no tuned score for {case_id} -->")
            continue
        c = cases[case_id]
        base = (c.get("base") or {}).get("test")
        table.append(dict(id=case_id, task=c["task"], test=c["test"], base=base,
                          duplicate=c["plan"] == "continuous_schedules_gauss512", **features(result)))
    print("| task | n | feature | rho vs tuned test | rho vs untuned test |")
    print("|---|---|---|---|---|")
    for task in sorted({r["task"] for r in table}):
        # one copy of the twice-trained Gaussian-512 models, as in the post
        rows = [r for r in table if r["task"] == task and not r["duplicate"]]
        sign = -1 if data["meta"]["tasks"][task]["dir"] == "min" else 1
        for f in ("gap", "lik", "b_low"):
            fx = [r[f] for r in rows]
            tuned = spearman(fx, [sign * r["test"] for r in rows])
            with_base = [r for r in rows if r["base"] is not None]
            untuned = spearman([r[f] for r in with_base], [sign * r["base"] for r in with_base]) if len(with_base) > 3 else float("nan")
            print(f"| {task} | {len(rows)} | {f} | {tuned:+.2f} | {untuned:+.2f} (n={len(with_base)}) |")
    print()
    print("| case | test | untuned | gap | lik | b_low |")
    print("|---|---|---|---|---|---|")
    for r in sorted(table, key=lambda r: (r["task"], -r["test"])):
        base = "" if r["base"] is None else f"{r['base']:.3f}"
        print(f"| {r['id']}{' (dup)' if r['duplicate'] else ''} | {r['test']:.3f} | {base} | {r['gap']:.3f} | {r['lik']:.3f} | {r['b_low']:.3f} |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
