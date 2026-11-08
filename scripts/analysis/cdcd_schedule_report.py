"""Learned CDCD schedules and loss-linearization check for continuous recipes.

Read-only over checkpoints. For every (task, path, feedback) case it writes:

- schedules.csv: global time u -> physical time t, the path's native noise
  parameter, and the CDCD fitted loss, on a fine grid.
- loss_bins.csv: measured validation cross entropy in equal-width bins of u
  next to the CDCD fit. CDCD samples t = F^-1(u) where F is a monotone fit of
  expected loss, so a working clock makes measured loss linear in u.
- metadata.json: checkpoint paths, steps, settings.

Run on the server through Ray Jobs; each case runs in a one-GPU Ray task:

    candi-remote submit --name cdcd-schedules -- python scripts/analysis/cdcd_schedule_report.py

Mirrors the earlier gaussian-loss-bins measurement (raw training weights,
training-mode SC policy, no gradients, no sampler updates).
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
TASKS = ["sudoku_hard", "text8", "tinystories"]
PATHS = ["gaussian_embedding", "gaussian_embedding_masked", "simplex", "spherical", "spherical_vmf"]
VARIANTS = ["none", "expected_embedding"]


def native(schedule, t):
    """Return {name: tensor} for the path's own noise parameter(s) at time t."""
    out = schedule(t)
    if hasattr(out, "sigma"):  # GaussianPath
        return {"sigma": out.sigma}
    if hasattr(out, "concentration"):  # SimplexPath
        return {"alpha": out.alpha, "concentration": out.concentration}
    kind = getattr(schedule, "output_kind", "value")  # spherical keep_rate / vMF concentration
    return {"keep_rate" if kind == "keep_rate" else "kappa": out[0]}


@ray.remote(num_gpus=1)
def run_case(task_name, path, variant, bins, examples, grid, split="validation"):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    torch.set_num_threads(4)
    case = ROOT / f"{task_name}-{path}-{variant}-seed-1"
    ckpts = sorted(case.glob("training/candidates/*/training/checkpoints/last.ckpt"))
    if len(ckpts) != 1:
        return {"case": case.name, "error": f"expected one last.ckpt, found {len(ckpts)}"}
    cfg = OmegaConf.load(case / "config.yaml")
    ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    algo = instantiate(cfg.algorithm)
    algo.load_state_dict(
        {k.removeprefix("algorithm."): v for k, v in ck["state_dict"].items() if k.startswith("algorithm.")},
        strict=True,
    )
    algo.cuda().train()
    clock = algo.time_sampler
    schedule = algo.model.views[0].schedule
    total = float(clock.ema_logits[1].float().exp().sum())

    # 1. Schedule curve on a fine grid of global time u.
    sched_rows = []
    with torch.no_grad():
        u = torch.linspace(0, 1, grid, device="cuda")
        t = torch.where((u == 0) | (u == 1), u, clock(u)).clamp(0, 1)
        fit = clock.cdf(t) * total
        params = {k: v.detach().float().cpu() for k, v in native(schedule, t).items()}
        for i in range(grid):
            row = {"task": task_name, "path": path, "variant": variant, "u": float(u[i]), "t": float(t[i]), "fitted_loss": float(fit[i])}
            row.update({k: float(v[i]) for k, v in params.items()})
            sched_rows.append(row)

    # 2. Measured loss in equal-width u bins.
    task = instantiate(cfg.task)
    ds = task.training_dataset() if split == "training" else task.validation_dataset()
    loader = DataLoader(Subset(ds, list(range(examples))), batch_size=16, collate_fn=task.collate_training)
    captured = []
    hook = algo.objective.register_forward_hook(lambda m, a, r: captured.append(r.detach()))
    original = clock.uniform.sample
    bin_rows = []
    try:
        for b in range(bins):
            values, times, fits = [], [], []
            for idx, batch in enumerate(loader):
                seed = 777 + b * 100 + idx
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                tokens = batch["input_ids"].cuda()
                mask = batch["loss_mask"].cuda()
                cm = batch["corruption_mask"].cuda()
                ub = (b + torch.rand(len(tokens), device="cuda")) / bins
                clock.uniform.sample = lambda count, device, _u=ub, **kw: _u
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    algo.loss(tokens, mask, corruption_mask=cm, training=False, step=int(ck["global_step"]))
                per = captured.pop()
                seq = torch.where(mask, per, 0).float().sum(-1) / mask.sum(-1).clamp_min(1)
                tb = clock(ub)
                values += seq.cpu().tolist()
                times += tb.cpu().tolist()
                fits += (clock.cdf(tb) * total).cpu().tolist()
            v = torch.tensor(values)
            bin_rows.append({
                "task": task_name, "path": path, "variant": variant, "u": (b + 0.5) / bins,
                "mean_t": sum(times) / len(times), "loss": float(v.mean()),
                "se": float(v.std() / math.sqrt(len(v))), "fitted_loss": sum(fits) / len(fits), "n": len(v),
            })
    finally:
        hook.remove()
        clock.uniform.sample = original

    meta = {
        "case": case.name, "task": task_name, "path": path, "variant": variant, "checkpoint": str(ckpts[0]),
        "step": int(ck["global_step"]), "ready": bool(clock.ready), "fitted_total": total,
        "schedule": type(schedule).__name__, "weights": "raw training weights (not EMA)",
        "mode": "training-mode SC policy; no gradients or sampler updates",
        "split": split, "examples": f"first {examples} {split} examples reused across {bins} equal-width u bins",
    }
    return {"case": case.name, "schedule": sched_rows, "bins": bin_rows, "meta": meta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(STORAGE / "analysis/cdcd-schedules-20261101"))
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--examples", type=int, default=128)
    p.add_argument("--grid", type=int, default=401)
    p.add_argument("--split", choices=["validation", "training"], default="validation")
    p.add_argument("--only", nargs="*", help="case-name substrings to restrict to (for a quick test)")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    ray.init(address="auto")
    jobs = [
        (t, pa, v) for t in TASKS for pa in PATHS for v in VARIANTS
        if not args.only or any(s in f"{t}-{pa}-{v}" for s in args.only)
    ]
    results = ray.get([run_case.remote(*j, args.bins, args.examples, args.grid, args.split) for j in jobs])

    sched, bins, meta, errors = [], [], [], []
    for r in results:
        if "error" in r:
            errors.append(r)
            print("ERROR", r, flush=True)
            continue
        sched += r["schedule"]
        bins += r["bins"]
        meta.append(r["meta"])
        print("done", r["case"], flush=True)

    keys = sorted({k for row in sched for k in row}, key=lambda k: (k not in ("task", "path", "variant", "u", "t", "fitted_loss"), k))
    with (out / "schedules.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sched)
    with (out / "loss_bins.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(bins[0]))
        w.writeheader()
        w.writerows(bins)
    (out / "metadata.json").write_text(json.dumps({
        "source_id": os.environ.get("CANDI_SOURCE_ID"), "cases": meta, "errors": errors,
        "settings": vars(args),
    }, indent=2))
    print(f"wrote {out}: {len(meta)} cases, {len(errors)} errors")


if __name__ == "__main__":
    main()
