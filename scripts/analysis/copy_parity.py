"""How copy-like or parity-like is a dataset? Plug-in information measures, orders 2 and 3.

For a target position i and two other positions j, k:
    pair info   I(i;j), I(i;k)
    joint info  I(i;jk)
    synergy index  g = I(i;jk) - I(i;j) - I(i;k)   (co-information gap)
g < 0 for copies (j and k each tell the same thing), g > 0 for a parity (only the pair
together tells anything), 0 when j and k carry separate pieces or one is unrelated.
With S = sum of positive gaps and R = sum of negative gaps (as bits) over local
triplets, the index (S - R) / (S + R) is -1 for pure copies and +1 for pure parities.
Every term is bias-corrected by subtracting the same estimate with position i shuffled
(true value 0). Orders above 3 are invisible here: a Sudoku row's
ninth cell is fixed by the other eight, which is a high-order (parity-like) dependence.

Usage (numpy + datasets):
    uv run --no-project --with numpy --with datasets python copy_parity.py --selfcheck
    uv run --no-project --with numpy --with datasets python copy_parity.py all --out OUT.json
"""

import argparse
import json
import math
import random
import sys
import zipfile
from pathlib import Path

import numpy as np

CANDI = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(0)


def entropy(codes):
    counts = np.bincount(codes)
    p = counts[counts > 0] / len(codes)
    return float(-(p * np.log2(p)).sum())


def info_terms(x, i, j, k, v):
    """Plug-in I(i;j), I(i;k), I(i;jk) in bits for integer columns of x (values < v)."""
    a, b, c = (x[:, t].astype(np.int64) for t in (i, j, k))
    h_a = entropy(a)
    mi = lambda y, vy: h_a + entropy(y) - entropy(a * vy + y)
    return mi(b, v), mi(c, v), mi(b * v + c, v * v)


def corrected_terms(x, i, j, k, v):
    raw = np.array(info_terms(x, i, j, k, v))
    shuffled = x.copy()
    shuffled[:, i] = RNG.permutation(shuffled[:, i])  # true value 0, same (j, k) table support
    return raw - np.array(info_terms(shuffled, i, j, k, v))


def context_info(x, i, context, v):
    """Plug-in I(x_i; x_context) in bits, bias-corrected by shuffling x_i."""
    joint = np.zeros(len(x), dtype=np.int64)
    for t in context:
        joint = joint * v + x[:, t]
    h = lambda codes: entropy(np.unique(codes, return_inverse=True)[1])
    a = x[:, i].astype(np.int64)
    raw = h(a) + h(joint) - h(joint * v + a)
    s = RNG.permutation(a)
    return raw - (h(s) + h(joint) - h(joint * v + s))


def info_curve(x, i, order, v, max_cells=2e7):
    """I(x_i; first m positions of `order`) for m = 1, 2, ... while v^(m+1) <= max_cells."""
    out = []
    for m in range(1, len(order) + 1):
        if v ** (m + 1) > max_cells:
            break
        out.append(round(context_info(x, i, order[:m], v), 4))
    return out


def summarize(x, triplets, v):
    """Net synergy S and net redundancy R summed over triplets; index (S - R) / (S + R).

    Per triplet the co-information gap g = I(i;jk) - I(i;j) - I(i;k) is positive for
    synergy (parity) and negative for redundancy (copies). A triplet with one unrelated
    position has g = 0, so it adds to neither sum."""
    r = np.array([corrected_terms(x, i, j, k, v) for i, j, k in triplets])
    gap = r[:, 2] - r[:, 0] - r[:, 1]
    syn, red = float(np.clip(gap, 0, None).sum()), float(np.clip(-gap, 0, None).sum())
    return dict(
        triplets=len(triplets),
        best_pair_info=float(r[:, :2].max()),
        best_joint_info=float(r[:, 2].max()),
        net_synergy_bits=syn,
        net_redundancy_bits=red,
        index=(syn - red) / (syn + red) if syn + red > 1e-3 else float("nan"),
    )


# ---------------------------------------------------------------- datasets

def windows(tokens, width, count):
    """count random windows of a long token stream, as rows."""
    starts = RNG.integers(0, len(tokens) - width, size=count)
    return np.stack([tokens[s:s + width] for s in starts])


def local_triplets(width, span):
    """Target in the middle of the window; j, k any two other positions within span."""
    i = width // 2
    others = [i + d for d in range(-span, span + 1) if d != 0]
    return [(i, j, k) for a, j in enumerate(others) for k in others[a + 1:]]


def text8(count):
    raw = CANDI / "core/data_cache/text8/raw/text8.zip"
    if not raw.exists():
        import urllib.request
        raw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://mattmahoney.net/dc/text8.zip", raw)
    text = zipfile.ZipFile(raw).read("text8")[:30_000_000]
    stream = np.frombuffer(text, dtype=np.uint8)
    used = np.unique(stream)
    lut = np.zeros(256, dtype=np.int64)
    lut[used] = np.arange(len(used))
    return lut[stream], len(used)


def tinystories(count):
    from datasets import load_dataset
    data = load_dataset("roneneldan/TinyStories", split="validation",
                        revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64")
    stream = "\n".join(data["text"]).encode("utf-8")
    used = sorted(set(stream))
    lut = np.zeros(256, dtype=np.int64)
    lut[used] = np.arange(len(used))
    return lut[np.frombuffer(stream, dtype=np.uint8)], len(used)


def sudoku_grids(count, seed=0):
    sys.path.insert(0, str(CANDI))
    from core.data.sudoku_generator import _fill_grid
    rng, grids = random.Random(seed), []
    for _ in range(count):
        grid = [[0] * 9 for _ in range(9)]
        _fill_grid(grid, rng)
        grids.append(sum(grid, []))
    return np.array(grids, dtype=np.int64) - 1  # values 0..8


def sudoku_triplets():
    """Target cell 40 (centre); pairs drawn from its row, column, box, or none of them."""
    i = 40
    r, c = divmod(i, 9)
    same = lambda t: t // 9 == r or t % 9 == c or (t // 27 == i // 27 and (t % 9) // 3 == c // 3)
    peers = [t for t in range(81) if t != i and same(t)]
    others = [t for t in range(81) if t != i and not same(t)]
    pick = lambda pool: [(i, j, k) for a, j in enumerate(pool) for k in pool[a + 1:]]
    return {"peers": pick(peers), "non_peers": pick(others[:20])}


def code_words(degree, count, n=64, r=16, locality=8):
    sys.path.insert(0, str(Path.home() / "candi-fgcode"))
    from core.data.factor_code import build_code
    _, g, _ = build_code(2, n, r, degree, locality, seed=0)
    return RNG.integers(0, 2, size=(count, r)) @ g % 2


# ---------------------------------------------------------------- runs

def nearest(i, span):
    """Positions ordered by distance from i, left before right: i-1, i+1, i-2, i+2, ..."""
    return [i + sgn * d for d in range(1, span + 1) for sgn in (-1, 1)]


def report(x, triplets, i, order, v):
    out = summarize(x, triplets, v)
    out["entropy_bits"] = round(entropy(x[:, i].astype(np.int64)), 4)
    out["info_vs_context"] = info_curve(x, i, order, v)
    return out


def run_stream(name, stream, v, width=17, span=7, count=2_000_000):
    x = windows(stream, width, count)
    i = width // 2
    return {name: report(x, local_triplets(width, span), i, nearest(i, span), v)}


def run_all():
    out = {}
    for degree in (2, 3):
        # window of 17 around position 32; checks span <= 8 positions, so span 7 sees them
        x = code_words(degree, 200_000)[:, 24:41]
        out[f"code_deg{degree}"] = report(x, local_triplets(17, 7), 8, nearest(8, 7), 2)
    grids = sudoku_grids(200_000)
    row = [36 + c for c in range(9) if c != 4]  # the other 8 cells of the centre cell's row
    for part, trips in sudoku_triplets().items():
        out[f"sudoku_{part}"] = report(grids, trips, 40, row, 9)
    for name, load in [("text8", text8), ("tinystories_bytes", tinystories)]:
        stream, v = load(0)
        out.update(run_stream(name, stream, v))
    return out


def selfcheck():
    n = 200_000
    a, b = RNG.integers(0, 2, n), RNG.integers(0, 2, n)
    copy = np.stack([a, a, a], 1)
    parity = np.stack([a ^ b, a, b], 1)
    noise = np.stack([a, a, b], 1)  # one copy and one unrelated position: no synergy, no redundancy
    assert math.isnan(summarize(noise, [(0, 1, 2)], 2)["index"])
    for name, x, v, want in [("copy", copy, 2, -1), ("parity", parity, 2, 1)]:
        s = summarize(x, [(0, 1, 2)], v)["index"]
        assert abs(s - want) < 0.02, (name, s)
        print(name, round(s, 3))
    print("selfcheck ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", nargs="?", default="all")
    p.add_argument("--out")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()
    if args.selfcheck:
        selfcheck()
        sys.exit()
    result = run_all()
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
