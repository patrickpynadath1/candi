#!/usr/bin/env python3
"""Ground-truth bits for the synthetic linear-code test bed.

A linear code over GF(q) (q prime), data = uniform random codeword. Each position goes
through one memoryless channel (erasure / q-ary symmetric flip / Gaussian on an embedding)
tuned to capacity c = I(X_i; S_i). We report, per position and averaged (bits):

  bits_channel  = H(X_i) - E[-log2 p(x_i | s_i)]          (context-free, should equal c)
  bits_exact    = H(X_i) - E[-log2 p(x_i | s)]            (ideal decoder, full codebook)
  bits_bp[T]    = H(X_i) - E[-log2 b_T(x_i | s)]          (sum-product BP, T rounds)

Capacities on the CLI are fractions of log2 q (c_bits = frac * log2 q).

  python code_truth.py --make 2 64 16 6 --channel flip --capacities 0.1,0.5,0.9 --out r.json
  python code_truth.py --code code.npz --channel erasure --out r.json
  python code_truth.py --selfcheck
Needs only numpy (e.g. `uv run --no-project --with numpy python code_truth.py ...`).
"""
import argparse
import json
import math
import sys
import time

import warnings

import numpy as np

# numpy on macOS Accelerate raises spurious FP warnings from matmul; finiteness is asserted instead
warnings.filterwarnings("ignore", message=".*encountered in matmul")

LOG_FLOOR = -1e4  # stands in for log 0 (erasure mismatch); exp() of it is exactly 0
P_FLOOR = 1e-30   # floor on BP messages so logs stay finite


# ---------------------------------------------------------------- GF(q) linear algebra
def rref(A, q):
    A = np.array(A, dtype=np.int64) % q
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        A[[r, p]] = A[[p, r]]
        A[r] = A[r] * pow(int(A[r, c]), q - 2, q) % q
        o = np.nonzero(A[:, c])[0]
        o = o[o != r]
        A[o] = (A[o] - np.outer(A[o, c], A[r])) % q
        piv.append(c)
        r += 1
    return A[:r], piv


def rank(A, q):
    A = np.asarray(A)
    return len(rref(A, q)[1]) if A.size else 0


def nullspace(A, q, n):
    """Basis (rows) of {x in GF(q)^n : A x = 0}; identity on the free columns."""
    A = np.asarray(A).reshape(-1, n)
    R, piv = rref(A, q) if len(A) else (np.zeros((0, n), np.int64), [])
    free = [c for c in range(n) if c not in piv]
    N = np.zeros((len(free), n), np.int64)
    for t, f in enumerate(free):
        N[t, f] = 1
        for row, p in enumerate(piv):
            N[t, p] = (-R[row, f]) % q
    return N, np.array(free, dtype=np.int64)


# ---------------------------------------------------------------- codes
def make_code(q, n, r, k, w=None, seed=0):
    """Random checks of exact degree k (k=2: coefficients (1,-1), a copy constraint),
    support inside a window of w consecutive positions if w is set. A drawn check is
    kept only if it raises rank(H), so rank(H) = n - r at the end."""
    assert k >= 2 and (w is None or w >= k) and r <= n
    rng = np.random.default_rng(seed)
    rows, tries = [], 0
    while len(rows) < n - r:
        tries += 1
        if tries > 200_000:
            raise RuntimeError("could not reach rank n-r; try a larger window or fewer checks")
        lo, span = (int(rng.integers(0, n - w + 1)), w) if w else (0, n)
        h = np.zeros(n, np.int64)
        h[lo + rng.choice(span, k, replace=False)] = [1, q - 1] if k == 2 else rng.integers(1, q, k)
        if rank(np.array(rows + [h]), q) == len(rows) + 1:
            rows.append(h)
    H = np.array(rows, np.int64).reshape(n - r, n)
    G, info = nullspace(H, q, n)
    return dict(H=H, G=G, info_set=info, q=q, degree=k, locality=w)


def load_code(path, q):
    z = np.load(path)
    q = int(z["q"]) if "q" in z else q
    H = z["H"].astype(np.int64) % q
    n = H.shape[1]
    G = z["G"].astype(np.int64) % q if "G" in z else nullspace(H, q, n)[0]
    info = z["info_set"] if "info_set" in z else nullspace(H, q, n)[1]
    deg = int((H != 0).sum(1).max()) if len(H) else 0
    return dict(H=H, G=G, info_set=np.asarray(info), q=q, degree=deg, locality=None)


def check_code(code):
    H, G, q = code["H"], code["G"], code["q"]
    n, r = G.shape[1], G.shape[0]
    assert all(pow(a, q - 1, q) == 1 for a in range(1, q)), "q must be prime"
    assert rank(G, q) == r, "G rows not independent"
    assert rank(H, q) == n - r, f"rank(H)={rank(H, q)} != n-r={n - r}"
    assert not ((H @ G.T) % q).any(), "H G^T != 0"
    info = np.asarray(code["info_set"])
    assert len(info) == r and rank(G[:, info], q) == r, "info_set does not determine the codeword"
    assert q ** r <= 1 << 22, "codebook too large to enumerate"


def codebook(G, q):
    C = np.zeros((1, G.shape[1]), np.int8)
    for g in G:  # all u G, one generator row at a time
        C = np.concatenate([(C + a * g) % q for a in range(q)]).astype(np.int8)
    return C


# ---------------------------------------------------------------- channels
def channel_loglik(kind, param, x, q, rng, E=None):
    """log p(s_i | x_i = a) for all a, shape (B, n, q), shifted so the max over a is 0."""
    B, n = x.shape
    sym = np.arange(q)
    if kind == "erasure":
        keep = rng.random((B, n)) < param
        L = np.where(sym == x[..., None], 0.0, LOG_FLOOR)
        L[~keep] = 0.0
    elif kind == "flip":
        s = np.where(rng.random((B, n)) < param, x, rng.integers(0, q, (B, n)))
        ps, pd = param + (1 - param) / q, (1 - param) / q
        L = np.where(sym == s[..., None], math.log(ps), math.log(pd) if pd > 0 else LOG_FLOOR)
    elif kind == "gaussian":
        E = np.eye(q) if E is None else E
        s = param * E[x] + rng.standard_normal((B, n, E.shape[1]))
        L = param * s @ E.T - 0.5 * param ** 2 * (E ** 2).sum(1)  # -|s - aE|^2/2 + const
    else:
        raise ValueError(kind)
    L = L - L.max(-1, keepdims=True)
    return np.maximum(L, LOG_FLOOR)


def softmax(L):
    P = np.exp(L - L.max(-1, keepdims=True))
    return P / P.sum(-1, keepdims=True)


def capacity(kind, param, q, E=None, ref=None):
    """I(X;S) in bits for uniform X."""
    lq = math.log2(q)
    if kind == "erasure":
        return param * lq
    if kind == "flip":
        ps, pd = param + (1 - param) / q, (1 - param) / q
        return lq + ps * math.log2(ps) + ((q - 1) * pd * math.log2(pd) if pd > 0 else 0.0)
    E = np.eye(q) if E is None else E  # gaussian: Monte Carlo with fixed draws (monotone in alpha)
    x, z = ref
    s = param * E[x] + z
    L = param * s @ E.T - 0.5 * param ** 2 * (E ** 2).sum(1)
    L -= L.max(-1, keepdims=True)
    logp = L[np.arange(len(x)), x] - np.log(np.exp(L).sum(-1))
    return lq + logp.mean() / math.log(2)


def solve_param(kind, c_bits, q, E=None, n_ref=200_000, seed=12345):
    lq = math.log2(q)
    assert 0 <= c_bits <= lq
    if kind == "erasure":
        return c_bits / lq
    ref = None
    if kind == "gaussian":
        rng = np.random.default_rng(seed)
        d = q if E is None else E.shape[1]
        ref = (rng.integers(0, q, n_ref), rng.standard_normal((n_ref, d)))
    lo, hi = 0.0, 1.0 if kind == "flip" else 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if capacity(kind, mid, q, E, ref) < c_bits else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- decoders
def exact_posterior(L, C, q, chunk=1 << 15):
    """p(x_i = a | s) under p(x|s) ∝ 1[x in C] prod_i p(s_i|x_i), by enumerating C."""
    B, n, _ = L.shape
    Lf = L.reshape(B, n * q)
    mx = np.full(B, -np.inf)
    acc = np.zeros((B, n * q))
    sym = np.arange(q, dtype=np.int8)
    for st in range(0, len(C), chunk):
        oh = (C[st:st + chunk, :, None] == sym).reshape(-1, n * q).astype(np.float64)
        S = Lf @ oh.T
        new = np.maximum(mx, S.max(1))
        acc = acc * np.exp(mx - new)[:, None] + np.exp(S - new[:, None]) @ oh
        mx = new
    acc = acc.reshape(B, n, q)
    return acc / acc.sum(-1, keepdims=True)


class BP:
    """Flooding sum-product on the Tanner graph of H over GF(q). Check update is a
    convolution over Z_q done with a length-q DFT (Hadamard for q=2)."""

    def __init__(self, H, q):
        self.q, self.n = q, H.shape[1]
        m = H.shape[0]
        deg = (H != 0).sum(1)
        d = int(deg.max()) if m else 0
        self.m, self.d = m, d
        var = np.zeros((m, d), np.int64)
        coef = np.ones((m, d), np.int64)
        mask = np.zeros((m, d), bool)
        for j in range(m):
            nz = np.nonzero(H[j])[0]
            var[j, :len(nz)], coef[j, :len(nz)], mask[j, :len(nz)] = nz, H[j, nz], True
        inv = np.array([pow(int(h), q - 2, q) for h in coef.ravel()]).reshape(m, d)
        a = np.arange(q)
        self.var, self.mask = var, mask
        self.inv_perm = (inv[..., None] * a) % q        # P_y[y] = P_x[h^-1 y], y = h x
        self.out_perm = (-coef[..., None] * a) % q      # P_x[a] = P_S[-h a]
        self.A = np.zeros((m * d, self.n))              # edge -> variable incidence
        self.A[np.arange(m * d)[mask.ravel()], var.ravel()[mask.ravel()]] = 1.0

    def beliefs(self, L, rounds):
        """{T: b_T(x_i = a | s)} for T in rounds."""
        out = {}
        if self.m == 0:
            return {T: softmax(L) for T in rounds}
        B, m, d, q = L.shape[0], self.m, self.d, self.q
        v2c = softmax(L)[:, self.var]                    # (B, m, d, q)
        for t in range(1, max(rounds) + 1):
            Py = np.take_along_axis(v2c, self.inv_perm[None], -1)
            F = np.fft.fft(Py, axis=-1)
            F[:, ~self.mask] = 1.0                       # padded slots: point mass at 0
            one = np.ones_like(F[:, :, :1])
            pre = np.cumprod(np.concatenate([one, F[:, :, :-1]], 2), 2)
            suf = np.cumprod(np.concatenate([one, F[:, :, :0:-1]], 2), 2)[:, :, ::-1]
            PS = np.fft.ifft(pre * suf, axis=-1).real
            c2v = np.maximum(np.take_along_axis(PS, self.out_perm[None], -1), P_FLOOR)
            lc = np.log(c2v / c2v.sum(-1, keepdims=True)) * self.mask[None, :, :, None]
            total = L + np.einsum("beq,en->bnq", lc.reshape(B, m * d, q), self.A)
            if t in rounds:
                out[t] = softmax(total)
            v2c = softmax(total[:, self.var] - lc)
        return out


# ---------------------------------------------------------------- estimation
def stats(loss, Hx):
    """loss: (N, n) per-sample -log2 p(true symbol). Returns mean/se/per-position bits."""
    bits = Hx[None] - loss
    per_sample = bits.mean(1)
    return dict(mean=float(per_sample.mean()),
                se=float(per_sample.std(ddof=1) / math.sqrt(len(per_sample))),
                per_position=[round(float(v), 5) for v in bits.mean(0)])


def estimate(code, kind, c_frac, rounds, samples=1000, batch=250, seed=0, E=None,
             keep_raw=False):
    q, G = code["q"], code["G"]
    if "_C" not in code:
        code["_C"], code["_bp"] = codebook(G, q), BP(code["H"], q)
    C, bp = code["_C"], code["_bp"]
    n = G.shape[1]
    Hx = np.where(G.any(0), math.log2(q), 0.0)  # uniform iff column of G nonzero
    c_bits = c_frac * math.log2(q)
    param = solve_param(kind, c_bits, q, E)
    rng = np.random.default_rng(seed)
    loss = {"channel": [], "exact": []} | {T: [] for T in rounds}
    raw = {"L": [], "x": []}
    for st in range(0, samples, batch):
        B = min(batch, samples - st)
        x = C[rng.integers(0, len(C), B)].astype(np.int64)
        L = channel_loglik(kind, param, x, q, rng, E)
        nl = lambda P: -np.log2(np.maximum(np.take_along_axis(P, x[..., None], -1)[..., 0], 1e-300))
        Lc = L.copy()
        Lc[:, Hx == 0, 1:] = LOG_FLOOR  # prior: a zero column of G means x_i is always 0
        loss["channel"].append(nl(softmax(Lc)))
        loss["exact"].append(nl(exact_posterior(L, C, q)))
        for T, P in bp.beliefs(L, rounds).items():
            loss[T].append(nl(P))
        if keep_raw:
            raw["L"].append(L), raw["x"].append(x)
    loss = {k: np.concatenate(v) for k, v in loss.items()}
    assert all(np.isfinite(v).all() for v in loss.values()), "non-finite loss"
    res = dict(c_frac=c_frac, c_bits=c_bits, param=param,
               param_name={"erasure": "keep_prob", "flip": "keep_prob", "gaussian": "alpha (sigma=1)"}[kind],
               bits_channel=stats(loss["channel"], Hx), bits_exact=stats(loss["exact"], Hx),
               bits_bp={str(T): stats(loss[T], Hx) for T in rounds})
    if keep_raw:
        res["_loss"], res["_Hx"] = loss, Hx
        res["_L"], res["_x"] = np.concatenate(raw["L"]), np.concatenate(raw["x"])
    return res


def erasure_span_bits(G, q, keep):
    """Closed form for erasure: x_i is known given x_K iff every u with u G_K = 0 has
    u G_i = 0; then bits = log2 q, else x_i is uniform given x_K and bits = 0."""
    r, n = G.shape
    out = np.zeros(keep.shape)
    for b, K in enumerate(keep):
        N, _ = nullspace(G[:, K].T, q, r)  # u with u G_K = 0
        out[b] = np.where(((N @ G) % q).any(0), 0.0, math.log2(q)) * G.any(0)
    return out


# ---------------------------------------------------------------- self-checks
def selfcheck():
    t0 = time.time()
    R = [1, 2, 4, 8, 16, 32]
    ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
    allok = True

    # 1. bits_channel ~ c; 2. exact >= bp >= channel (paired, within MC error)
    codes = [make_code(2, 24, 10, 3, seed=1), make_code(3, 16, 7, 3, seed=2),
             make_code(5, 10, 5, 3, seed=3), make_code(7, 8, 4, 2, seed=4)]
    for code in codes:
        check_code(code)
        q = code["q"]
        for kind in ["erasure", "flip", "gaussian"]:
            for cf in [0.2, 0.5, 0.8]:
                r = estimate(code, kind, cf, R, samples=600, seed=7, keep_raw=True)
                bc, lq = r["bits_channel"], math.log2(q)
                c_exp = r["c_bits"] * (r["_Hx"] > 0).mean()  # constant positions (zero G column) carry 0
                allok &= ok(abs(bc["mean"] - c_exp) < 4 * bc["se"] + 0.005 * lq,
                            f"q={q} {kind:8s} c={cf}: bits_channel {bc['mean']:.4f} vs c {c_exp:.4f} (se {bc['se']:.4f})")
                lo = r["_loss"]
                for T in R:
                    for hi_k, lo_k, name in [("exact", T, "exact>=bp"), (T, "channel", "bp>=channel")]:
                        diff = (lo[lo_k] - lo[hi_k]).mean(1)  # bits(hi) - bits(lo)
                        se = diff.std(ddof=1) / math.sqrt(len(diff))
                        if not diff.mean() > -4 * se - 1e-9:
                            allok &= ok(False, f"q={q} {kind} c={cf} T={T}: {name} off by {diff.mean():.4f} (se {se:.4f})")

                # 3. erasure: exact == linear-algebra span check, per sample and position
                if kind == "erasure":
                    keep = (r["_L"] < -1).any(-1)
                    span = erasure_span_bits(code["G"], q, keep)
                    exact = r["_Hx"][None] - lo["exact"]
                    allok &= ok(np.abs(span - exact).max() < 1e-6,
                                f"q={q} erasure c={cf}: exact == span (max err {np.abs(span - exact).max():.1e})")
    print("checked exact >= bp(T) >= channel on all of the above (FAIL lines only)")

    # 4. copy code (degree 2) is a forest: BP with enough rounds is exact, any channel
    for q in [2, 3]:
        code = make_code(q, 24, 8, 2, seed=5)
        check_code(code)
        for kind in ["erasure", "flip", "gaussian"]:
            r = estimate(code, kind, 0.5, [32], samples=300, seed=8, keep_raw=True)
            err = np.abs(r["_loss"][32] - r["_loss"]["exact"]).max()
            allok &= ok(err < 1e-5, f"copy code q={q} {kind}: BP(32) == exact per sample (max err {err:.1e})")

    # 5. no checks (r = n): exact == channel
    code = make_code(2, 10, 10, 2, seed=6)
    check_code(code)
    for kind in ["erasure", "flip", "gaussian"]:
        r = estimate(code, kind, 0.5, [4], samples=300, seed=9, keep_raw=True)
        err = np.abs(r["_loss"]["exact"] - r["_loss"]["channel"]).max()
        allok &= ok(err < 1e-8, f"r=n {kind}: exact == channel (max err {err:.1e})")

    print(f"{'ALL PASS' if allok else 'SOME FAILED'} in {time.time() - t0:.0f}s")
    assert allok


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--code", help="code.npz with H, G, info_set (optional q)")
    g.add_argument("--make", nargs="+", type=int, metavar="q n r degree [locality]")
    ap.add_argument("--q", type=int, default=2, help="field size if code.npz has no q")
    ap.add_argument("--channel", choices=["erasure", "flip", "gaussian"])
    ap.add_argument("--embed", help=".npy embedding table E (q x d) for --channel gaussian")
    ap.add_argument("--capacities", default=",".join(f"{c:.2f}" for c in np.arange(0.02, 0.99, 0.04)),
                    help="fractions of log2 q")
    ap.add_argument("--rounds", default="1,2,4,8,16,32")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if not (a.code or a.make) or not a.channel:
        ap.error("need --code or --make, and --channel")
    if a.make:
        q, n, r, k = a.make[:4]
        code = make_code(q, n, r, k, a.make[4] if len(a.make) > 4 else None, seed=a.seed)
    else:
        code = load_code(a.code, a.q)
    check_code(code)
    E = np.load(a.embed) if a.embed else None
    if E is not None:
        assert E.shape[0] == code["q"], "embedding table must have q rows"
    rounds = [int(t) for t in a.rounds.split(",")]
    q, (r, n) = code["q"], code["G"].shape
    Hx = np.where(code["G"].any(0), math.log2(q), 0.0)
    meta = dict(q=q, n=n, r=r, degree=code["degree"], locality=code["locality"], channel=a.channel,
                H_X=float(Hx.mean()), zero_columns=int((Hx == 0).sum()), samples=a.samples, seed=a.seed,
                source=a.code or f"make {a.make}", capacity_units="fraction of log2 q",
                embed=a.embed, rounds=rounds)
    results = []
    for cf in [float(c) for c in a.capacities.split(",")]:
        t0 = time.time()
        res = estimate(code, a.channel, cf, rounds, samples=a.samples, seed=a.seed, E=E)
        results.append(res)
        bp = " ".join(f"T{T}={res['bits_bp'][str(T)]['mean']:.3f}" for T in rounds)
        print(f"c={cf:.2f} chan={res['bits_channel']['mean']:.3f} exact={res['bits_exact']['mean']:.3f}"
              f"±{res['bits_exact']['se']:.3f} {bp} ({time.time() - t0:.1f}s)", file=sys.stderr)
    out = json.dumps(dict(meta=meta, results=results), indent=1)
    if a.out:
        open(a.out, "w").write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
