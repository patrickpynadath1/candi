"""Learned CDCD schedule and loss-linearity check for one trained case.

Read-only over a checkpoint. For a continuous recipe trained with the CDCD
learned clock it measures:

- the schedule: global time u -> physical time t = clock(u), the path's own
  noise parameter at t (Gaussian sigma, spherical keep rate, vMF kappa), and
  the normalized CDCD fit F(t) / F(1), on a grid of u;
- loss bins: measured sequence cross entropy in equal-width bins of u, next
  to the CDCD fit. CDCD samples t = F^-1(u) where F is a monotone fit of the
  expected loss, so a working clock makes the measured loss linear in u.

Settings follow the published measurement: raw training weights (not EMA),
the training-mode SC policy, no gradients and no sampler updates, the first
``examples`` examples of the split reused in every bin, fixed seeds per bin.
Losses are reported in units of the fitted total F(1) (``*_nats`` keep nats).
"""

import json
import math
from pathlib import Path


def load_trained(trained, device="cpu"):
    """(config, checkpoint, algorithm) for a ``trained.json`` record or path."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from core.training.checkpoints import algorithm_state

    if not isinstance(trained, dict):
        trained = json.loads(Path(trained).read_text())
    config = OmegaConf.load(trained["config"])
    checkpoint = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
    algorithm = instantiate(config.algorithm)
    algorithm.load_state_dict(algorithm_state(checkpoint), strict=True)
    return config, checkpoint, algorithm.to(device)


def native(schedule, t):
    """{name: tensor} for the path's own noise parameter at time t (none for simplex)."""
    out = schedule(t)
    if hasattr(out, "sigma"):  # Gaussian path
        return {"sigma": out.sigma}
    if hasattr(out, "concentration"):  # simplex: not shown in the published panels
        return {}
    kind = getattr(schedule, "output_kind", "value")  # spherical keep rate / vMF concentration
    return {"keep_rate" if kind == "keep_rate" else "kappa": out[0]}


def _round(value, digits=5):
    return round(float(value), digits)


def schedule_curve(algorithm, grid=101):
    """{u, t, fit, <native parameter>} on ``grid`` equally spaced u in [0, 1]."""
    import torch

    clock, schedule = algorithm.time_sampler, algorithm.views[0].schedule
    device = clock.ema_logits.device
    with torch.no_grad():
        u = torch.linspace(0, 1, grid, device=device)
        t = torch.where((u == 0) | (u == 1), u, clock(u)).clamp(0, 1)
        curve = {"u": u, "t": t, "fit": clock.cdf(t)}
        curve.update(native(schedule, t))
    return {key: [_round(v) for v in value.float().cpu().tolist()] for key, value in curve.items()}


def loss_bins(algorithm, config, step, *, split="validation", bins=20, examples=128, batch_size=16):
    """Measured loss per equal-width u bin, normalized by the fitted total F(1)."""
    import torch
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader, Subset

    clock = algorithm.time_sampler
    device = clock.ema_logits.device
    total = float(clock.ema_logits[1].float().exp().sum())
    task = instantiate(config.task)
    dataset = task.training_dataset() if split == "training" else task.validation_dataset()
    count = min(examples, len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(count))), batch_size=batch_size, collate_fn=task.collate_training
    )
    captured = []
    hook = algorithm.objective.register_forward_hook(lambda module, args, out: captured.append(out.detach()))
    original = clock.uniform.sample
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    algorithm.train()
    rows = []
    try:
        for b in range(bins):
            values, fits = [], []
            for index, batch in enumerate(loader):
                seed = 777 + b * 100 + index
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                tokens = batch["input_ids"].to(device)
                mask = batch["loss_mask"].to(device)
                corruption = batch.get("corruption_mask", batch["loss_mask"]).to(device)
                ub = (b + torch.rand(len(tokens), device=device)) / bins
                clock.uniform.sample = lambda count, device, _u=ub, **context: _u
                with torch.no_grad(), autocast:
                    algorithm.loss(tokens, mask, corruption_mask=corruption, training=False, step=step)
                per_token = captured.pop()
                sequence = torch.where(mask, per_token, 0).float().sum(-1) / mask.sum(-1).clamp_min(1)
                values += sequence.cpu().tolist()
                fits += clock.cdf(clock(ub)).cpu().tolist()
            v = torch.tensor(values, dtype=torch.float64)
            se = float(v.std() / math.sqrt(len(v))) if len(v) > 1 else 0.0
            mean = float(v.mean())
            rows.append({
                "u": _round((b + 0.5) / bins), "loss": _round(mean / total), "se": _round(se / total),
                "fit": _round(sum(fits) / len(fits)), "std": _round(se * math.sqrt(len(v)) / total),
                "std_nats": _round(se * math.sqrt(len(v))), "loss_nats": _round(mean),
            })
    finally:
        hook.remove()
        clock.uniform.sample = original
    return rows


def linearity(rows):
    """R^2 of the least-squares line through measured loss against u (NaN if flat)."""
    x = [row["u"] for row in rows]
    y = [row["loss"] for row in rows]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy * sxy / (sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def schedule_case(trained, *, bins=20, examples=128, grid=101, device="cpu"):
    """Schedule curve, validation and training loss bins, and metadata for one case."""
    if not isinstance(trained, dict):
        trained = json.loads(Path(trained).read_text())
    config, checkpoint, algorithm = load_trained(trained, device)
    step = int(checkpoint["global_step"])
    clock = algorithm.time_sampler
    result = {
        "schedule": schedule_curve(algorithm, grid),
        "bins": loss_bins(algorithm, config, step, split="validation", bins=bins, examples=examples),
        "bins_train": loss_bins(algorithm, config, step, split="training", bins=bins, examples=examples),
    }
    result["meta"] = {
        "checkpoint": str(trained["checkpoint"]),
        "step": step,
        "ready": bool(clock.ready),
        "weights": "raw training weights (not EMA)",
        "mode": "training-mode SC policy; no gradients or sampler updates",
        "examples": f"first {examples} validation (and training) examples reused across {bins} equal-width u bins",
    }
    return result
