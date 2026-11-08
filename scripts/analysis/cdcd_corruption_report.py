"""Token corruption and rank degradation along learned CDCD schedules.

Read-only over checkpoints. For each pure continuous case (Gaussian embeddings,
simplex, spherical, vMF) x task x feedback, and each global time u:

1. t = learned CDCD clock(u); coefficients = the path's schedule at t.
2. Draw clean tokens uniformly over the vocabulary, corrupt them with the
   path's own forward kernel (process.corrupt_state) and learned codebook.
3. Decode: scores = noisy @ unit codebook (simplex: the noisy simplex point).
   - error: P(argmax score != clean token)   (discrete corruption)
   - normalized rank: one-based midrank of the clean token, (rank-1)/(V-1)
     (continuous rank degradation). Independent noise gives 1-1/V and 0.5.

For Gaussian paths it also evaluates the closed-form rank used by
core.noise_schedules.embedding_rank.embedding_table_rank on the learned table,
as a check on the Monte Carlo estimate.

Mirrors the earlier geometry-empirical / gaussian-text-empirical analysis, run
on the final 20k checkpoints. Submit through Ray Jobs:

    candi-remote submit --name cdcd-corruption -- python scripts/analysis/cdcd_corruption_report.py
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import ray

STORAGE = Path(os.environ.get("CANDI_STORAGE_ROOT", Path.home() / ".cache/candi"))
ROOT = STORAGE / "nested-tuning/feedback-depth-cdcd-20k-20261030"
TASKS = ["text8", "tinystories", "sudoku_hard"]
PATHS = ["gaussian_embedding", "simplex", "spherical", "spherical_vmf"]
VARIANTS = ["none", "expected_embedding"]
TABLE_KEY = "algorithm.model.views.0.representation.state_representation.table"


@ray.remote(num_gpus=1)
def run_case(task_name, path, variant, samples, points):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from core.noise_schedules.embedding_rank import embedding_table_rank

    torch.set_num_threads(4)
    case = ROOT / f"{task_name}-{path}-{variant}-seed-1"
    ckpts = sorted(case.glob("training/candidates/*/training/checkpoints/last.ckpt"))
    if len(ckpts) != 1:
        return {"case": case.name, "error": f"expected one last.ckpt, found {len(ckpts)}"}
    cfg = OmegaConf.load(case / "config.yaml")
    ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    algo = instantiate(cfg.algorithm)
    algo.load_state_dict({k.removeprefix("algorithm."): v for k, v in state.items() if k.startswith("algorithm.")}, strict=True)
    algo.cuda().eval()
    clock, view = algo.time_sampler, algo.model.views[0]
    V = int(cfg.task.vocab_size)
    book = None
    if path != "simplex":
        book = torch.nn.functional.normalize(state[TABLE_KEY].float().cuda(), dim=-1)
        assert book.shape[0] == V, (book.shape, V)

    u = torch.unique(torch.cat([torch.linspace(0, 1, points), torch.tensor([.002, .005, .01, .02, .99, .995])])).cuda()
    rows = []
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    tokens = torch.randint(V, (samples, 1), device="cuda")
    mask = torch.ones_like(tokens, dtype=torch.bool)
    with torch.no_grad():
        t = torch.where((u == 0) | (u == 1), u, clock(u)).clamp(0, 1)
        for a, b in zip(u, t):
            coefficients = view.schedule(b.expand(samples))
            clean = view.representation.encode(tokens) if book is None else book[tokens]
            noisy = view.process.corrupt_state(clean, coefficients, mask)
            scores = noisy[:, 0].float() if book is None else noisy[:, 0].float() @ book.T
            target = scores.gather(1, tokens)
            ranks = (scores > target).sum(-1).float() + 0.5 * ((scores == target).sum(-1) - 1).float()
            err = (scores.argmax(-1) != tokens[:, 0]).float()
            nr = ranks / (V - 1)
            row = {
                "task": task_name, "path": path, "variant": variant, "u": float(a), "t": float(b),
                "error": float(err.mean()), "error_se": float(err.std() / math.sqrt(samples)),
                "rank": float(nr.mean()), "rank_se": float(nr.std() / math.sqrt(samples)),
            }
            sigma = getattr(coefficients, "sigma", None)
            if book is not None and sigma is not None:
                # embedding_table_rank uses z = (1-p) e + p N(0, I); GaussianPath has alpha = 1 - sigma.
                row["rank_formula"] = float(embedding_table_rank(book, sigma[0].clamp(1e-6, 1)))
            rows.append(row)
    meta = {
        "case": case.name, "checkpoint": str(ckpts[0]), "step": int(ck["global_step"]), "vocab_size": V,
        "samples_per_point": samples, "seed": 12345, "clean_tokens": "uniform over the full vocabulary",
        "decoder": "simplex: argmax of noisy simplex point; others: nearest unit learned codeword (dot product)",
        "rank": "one-based midrank; normalized (rank-1)/(V-1)", "weights": "raw checkpoint (not EMA)",
    }
    return {"case": case.name, "rows": rows, "meta": meta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(STORAGE / "analysis/cdcd-corruption-20261101"))
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--points", type=int, default=101)
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    ray.init(address="auto")
    jobs = [(t, pa, v) for t in TASKS for pa in PATHS for v in VARIANTS
            if not args.only or any(s in f"{t}-{pa}-{v}" for s in args.only)]
    results = ray.get([run_case.remote(*j, args.samples, args.points) for j in jobs])
    rows, meta, errors = [], [], []
    for r in results:
        if "error" in r:
            errors.append(r)
            print("ERROR", r, flush=True)
            continue
        rows += r["rows"]
        meta.append(r["meta"])
        print("done", r["case"], flush=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "corruption.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "metadata.json").write_text(json.dumps(
        {"source_id": os.environ.get("CANDI_SOURCE_ID"), "cases": meta, "errors": errors, "settings": vars(args)}, indent=2))
    print(f"wrote {out}: {len(meta)} cases, {len(errors)} errors")


if __name__ == "__main__":
    main()
