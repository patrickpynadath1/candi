"""Summarize the factorial sweep without selecting on the reporting puzzles.

One sampler is chosen per variant x SC using its mean selection score across all
planned training seeds. Paired per-seed held-out scores and all policies remain
available. This report does not compute significance from puzzle pseudoreplicates.
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from omegaconf import OmegaConf


TARGETS = {
    "core.samplers.hybrid.HybridSampler": "native_128",
    "core.samplers.parallel_entropy_bounded.ParallelEntropyBoundedSampler": "eb_greedy",
}


def summarize_records(records, protocol, policy):
    by_index = {r["index"]: r for r in records}
    expected = set(protocol["selection_indices"]) | set(protocol["report_indices"])
    if len(by_index) != len(records) or set(by_index) != expected:
        raise ValueError("Evaluation must contain each planned puzzle exactly once")
    if set(protocol["selection_indices"]) & set(protocol["report_indices"]):
        raise ValueError("Selection and reporting puzzles overlap")
    for r in records:
        if not isinstance(r["valid_solution"], bool):
            raise ValueError("valid_solution must be a boolean")
        nfe = r["nfe"]
        if not isinstance(nfe, (int, float)) or not math.isfinite(nfe):
            raise ValueError("Missing or nonfinite NFE")
        if policy.startswith("native_128") and nfe != 128:
            raise ValueError("Native evaluation did not use exactly 128 NFE")
        if not 0 < nfe <= 128:
            raise ValueError("Evaluation exceeded its 128-NFE budget")
    result = {}
    for label, indices in (
        ("selection", protocol["selection_indices"]),
        ("report", protocol["report_indices"]),
        ("all", sorted(expected)),
    ):
        if not indices:
            raise ValueError("Both selection and reporting splits must be nonempty")
        result[f"{label}_success_percent"] = 100 * mean(
            by_index[i]["valid_solution"] for i in indices
        )
        result[f"{label}_mean_nfe"] = mean(by_index[i]["nfe"] for i in indices)
    return result


def select_policies(rows, plan):
    """Return complete groups only; never maximize independently within each seed."""
    groups = {}
    for job in plan["jobs"]:
        f = job["config"]["factorial"]
        key = (f["variant"], f["self_conditioning"])
        groups.setdefault(key, []).append(f)
    indexed = {(r["run_name"], r["policy"]): r for r in rows}
    selected, incomplete = [], []
    for (variant, sc), factors in sorted(groups.items()):
        missing = [
            f"{f['run_name']}/{policy}"
            for f in factors
            for policy in plan["protocol"]["policies"]
            if (f["run_name"], policy) not in indexed
        ]
        if missing:
            incomplete.extend(missing)
            continue
        candidates = {}
        for policy in plan["protocol"]["policies"]:
            values = [indexed[f["run_name"], policy] for f in factors]
            candidates[policy] = (
                mean(r["selection_success_percent"] for r in values),
                mean(r["selection_mean_nfe"] for r in values),
            )
        best = min(candidates, key=lambda p: (-candidates[p][0], candidates[p][1], p))
        for f in factors:
            selected.append({**indexed[f["run_name"], best], "selected_policy": best})
    return selected, incomplete


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report(experiment):
    experiment = Path(experiment)
    plan = json.loads((experiment / "job-plan.json").read_text())
    planned = {j["name"]: j for j in plan["jobs"]}
    protocol = plan["protocol"]
    found, puzzle_targets = {}, None
    # Ray keeps one working run directory per trial; checkpoint copies are excluded.
    for path in sorted(experiment.glob("**/run/results.json")):
        cfg = OmegaConf.load(path.parent / "config.yaml")
        if cfg.get("mode") != "evaluate" or not cfg.get("factorial"):
            continue
        f = OmegaConf.to_container(cfg.factorial, resolve=True)
        if f.get("study") != "corruption_self_conditioning":
            continue
        name = f["run_name"]
        if name not in planned or f != planned[name]["config"]["factorial"]:
            raise ValueError(f"Unknown or changed factors in {path}")
        policy = TARGETS.get(cfg.sampler._target_)
        if policy == "native_128" and cfg.sampler.get("greedy_clean_estimate", False):
            policy = "native_128_greedy"
        if policy is None:
            raise ValueError(f"Unplanned sampler in {path}")
        expected_job = planned[name]
        policy_index = protocol["policies"].index(policy)
        choice = expected_job["inference"][policy_index]
        # Validate the resolved training configuration and all inference settings.
        from core.sweeps import compose

        expected_cfg = OmegaConf.create(expected_job["config"])
        expected_cfg.sampler = compose(overrides={"sampler": choice["sampler"]}).sampler
        for key, value in choice["overrides"].items():
            OmegaConf.update(expected_cfg, key, value, merge=False)
        expected_cfg.output_dir = str(path.parent.resolve())
        expected_cfg = OmegaConf.to_container(expected_cfg, resolve=True)
        actual = OmegaConf.to_container(cfg, resolve=True)
        for field in (
            "algorithm",
            "model",
            "objective",
            "time_sampler",
            "self_conditioning",
            "prediction",
            "parameterization",
            "data",
            "task",
            "optimizer",
            "training",
            "sampler",
            "seed",
            "loader",
            "block_diffusion",
            "token_feedback",
        ):
            if actual.get(field) != expected_cfg.get(field):
                raise ValueError(f"Changed {field} in {path}")
        if cfg.trainer.precision != expected_cfg["trainer"]["precision"]:
            raise ValueError(f"Changed precision in {path}")
        result = json.loads(path.read_text())
        if result.get("checkpoint_step") != protocol["train_steps"]:
            raise ValueError(f"Wrong checkpoint step in {path}")
        if not cfg.evaluation.use_ema:
            raise ValueError(f"Expected EMA weights in {path}")
        metrics = summarize_records(result["records"], protocol, policy)
        targets = {r["index"]: r["ground_truth"] for r in result["records"]}
        if puzzle_targets is not None and targets != puzzle_targets:
            raise ValueError(f"Evaluation puzzle targets changed in {path}")
        puzzle_targets = targets
        row = {
            "run_name": name,
            "family": f["family"],
            "variant": f["variant"],
            "gaussian_schedule": f["gaussian_schedule"],
            "schedule_adaptation": f["schedule_adaptation"],
            "self_conditioning": f["self_conditioning"],
            "training_seed": f["training_seed"],
            "policy": policy,
            **metrics,
            "result_file": str(path.resolve()),
        }
        key = (name, policy)
        if key in found:
            raise ValueError(
                f"Duplicate completed evaluation {key}; resolve retries explicitly"
            )
        found[key] = row
    rows = list(found.values())
    selected, incomplete = select_policies(rows, plan)
    output = experiment / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "all-policies.csv", rows)
    write_csv(output / "selected-per-seed.csv", selected)
    groups = {}
    for row in selected:
        groups.setdefault((row["variant"], row["self_conditioning"]), []).append(row)
    summary = []
    for (variant, sc), values in sorted(groups.items()):
        scores = [r["report_success_percent"] for r in values]
        summary.append(
            {
                "family": values[0]["family"],
                "variant": variant,
                "self_conditioning": sc,
                "selected_policy": values[0]["selected_policy"],
                "seeds": len(values),
                "selection_success_percent": mean(
                    r["selection_success_percent"] for r in values
                ),
                "report_success_percent": mean(scores),
                "report_seed_sd_pp": stdev(scores) if len(scores) > 1 else None,
                "report_mean_nfe": mean(r["report_mean_nfe"] for r in values),
            }
        )
    write_csv(output / "summary.csv", summary)
    (output / "status.json").write_text(
        json.dumps(
            {
                "complete": not incomplete,
                "completed_evaluations": len(rows),
                "expected_evaluations": len(protocol["policies"]) * len(planned),
                "missing": incomplete,
                "plan_fingerprint": plan["fingerprint"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Report: {output} ({len(rows)}/{len(protocol['policies']) * len(planned)} evaluations)"
    )
    if incomplete:
        print("Incomplete groups have no selected winner; see analysis/status.json.")
    return {
        "rows": rows,
        "selected": selected,
        "summary": summary,
        "missing": incomplete,
    }
