"""Resumable evaluation grids and task-independent Pareto summaries.

Each point runs the normal root entrypoint in a fresh process, so model/scorer
memory is released between points. Sampling, metrics and checkpoint loading
remain owned by their existing components.
"""

import csv
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

from core.evaluation.reporting import scalar_metrics


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pareto_flags(rows, objectives, group_by=()):
    """Equal points both survive; dominance requires one strict improvement."""

    def dominates(other, row):
        gains = [
            (other[key] - row[key]) * (1 if direction == "max" else -1)
            for key, direction in objectives.items()
        ]
        return all(gain >= 0 for gain in gains) and any(gain > 0 for gain in gains)

    for key, direction in objectives.items():
        if direction not in {"min", "max"}:
            raise ValueError(f"objective {key}: direction must be min or max")
        for row in rows:
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"objective {key!r} must be a finite scalar in every result"
                )
    for key in group_by:
        if any(key not in row for row in rows):
            raise ValueError(f"unknown frontier grouping field: {key}")
    if len(objectives) < 2:
        raise ValueError("a frontier requires at least two objectives")
    return [
        not any(
            all(other[key] == row[key] for key in group_by) and dominates(other, row)
            for other in rows
        )
        for row in rows
    ]


def plot_svg(path, rows, objectives):
    """Dependency-free two-objective scatter plot; full dimensions stay in CSV."""
    xkey, ykey = list(objectives)[:2]
    xmin, xmax = min(r[xkey] for r in rows), max(r[xkey] for r in rows)
    ymin, ymax = min(r[ykey] for r in rows), max(r[ykey] for r in rows)

    def x(value):
        return 90 + 570 * ((value - xmin) / (xmax - xmin) if xmax > xmin else 0.5)

    def y(value):
        return 380 - 320 * ((value - ymin) / (ymax - ymin) if ymax > ymin else 0.5)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" viewBox="0 0 760 470">',
        '<rect width="760" height="470" fill="white"/>',
        '<path d="M90 60V380H660" fill="none" stroke="black"/>',
    ]
    for row in rows:
        color = "#d34b28" if row["pareto_frontier"] else "#788b9a"
        label = html.escape(
            f"{row['point_id']}: {xkey}={row[xkey]:.5g}, {ykey}={row[ykey]:.5g}"
        )
        parts.append(
            f'<circle cx="{x(row[xkey]):.3f}" cy="{y(row[ykey]):.3f}" r="5" fill="{color}"><title>{label}</title></circle>'
        )
    labels = [
        (90, 405, f"{xmin:.4g}"),
        (620, 405, f"{xmax:.4g}"),
        (10, 380, f"{ymin:.4g}"),
        (10, 65, f"{ymax:.4g}"),
        (90, 440, f"{xkey} ({objectives[xkey]})"),
        (90, 25, f"{ykey} ({objectives[ykey]}); orange = nondominated within group"),
    ]
    for left, top, label in labels:
        parts.append(
            f'<text x="{left}" y="{top}" font-family="sans-serif" font-size="13">{html.escape(label)}</text>'
        )
    path.write_text("\n".join([*parts, "</svg>"]))


def summarize(output, rows, objectives, group_by):
    flags = pareto_flags(rows, objectives, group_by)
    rows = [dict(row, pareto_frontier=flag) for row, flag in zip(rows, flags)]
    write_json(
        output / "frontiers.json",
        {"objectives": objectives, "group_by": group_by, "points": rows},
    )
    fields = sorted({key for row in rows for key in row})
    with (output / "frontiers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        plot_svg(output / "frontier.svg", rows, objectives)
    return rows


def run(config, *, evaluation_entrypoint):
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    options = OmegaConf.to_container(config.frontier, resolve=True)
    base = OmegaConf.create(config)
    del base.frontier
    if base.mode not in {"evaluate", "sample"}:
        raise ValueError("frontier sweeps require mode=evaluate or mode=sample")
    axes = options["axes"]
    names = [axis["parameter"] for axis in axes]
    if len(names) != len(set(names)) or any(not axis["values"] for axis in axes):
        raise ValueError("sweep axes must be unique and contain at least one value")
    for name in names:
        if name in {"output_dir", "mode"} or name.startswith(("frontier.", "hydra.")):
            raise ValueError(f"cannot sweep orchestration setting {name}")
        if OmegaConf.select(base, name, default="__absent__") == "__absent__":
            raise ValueError(f"unknown sweep parameter: {name}")
    objectives = options["objectives"]
    # Validate the declarations even when planning without execution.
    pareto_flags([], objectives)
    seeds = options["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or "seed" in names:
        raise ValueError(
            "use a nonempty, unique frontier.seeds list instead of a seed axis"
        )
    plans = []
    for values in itertools.product(*(axis["values"] for axis in axes)):
        for seed in seeds:
            settings = OmegaConf.create(base)
            for name, value in zip(names, values):
                OmegaConf.update(settings, name, value, merge=False)
            settings.seed = seed
            # Resolve interpolations after axes, so model/predictor references follow overrides.
            resolved = OmegaConf.to_container(settings, resolve=True)
            resolved.pop("output_dir", None)
            checkpoint = Path(resolved["evaluation"]["checkpoint"] or "").resolve()
            if not checkpoint.is_file():
                raise ValueError(
                    f"evaluation.checkpoint must be an existing checkpoint: {checkpoint}"
                )
            resolved["evaluation"]["checkpoint"] = str(checkpoint)
            stat = checkpoint.stat()
            signature = {
                "config": resolved,
                "checkpoint_size": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns,
            }
            digest = hashlib.sha256(
                json.dumps(signature, sort_keys=True).encode()
            ).hexdigest()
            point = output / "points" / digest[:16]
            resolved["output_dir"] = str(point)
            plans.append(
                {
                    "point_id": digest[:16],
                    "signature": digest,
                    "axes": dict(zip(names, values)),
                    "seed": seed,
                    "directory": str(point),
                    "config": resolved,
                }
            )
    # Validate the whole grid before launching any GPU work.
    write_json(
        output / "manifest.json",
        {"objectives": objectives, "group_by": options["group_by"], "points": plans},
    )
    if options["dry_run"]:
        print(
            f"Planned {len(plans)} evaluations in {output / 'manifest.json'}",
            flush=True,
        )
        return []
    rows = []
    for index, plan in enumerate(plans):
        point = Path(plan["directory"])
        point.mkdir(parents=True, exist_ok=True)
        result_path = point / "results.json"
        marker = point / "completed.json"
        reusable = (
            options["resume"]
            and result_path.exists()
            and marker.exists()
            and json.loads(marker.read_text())["signature"] == plan["signature"]
        )
        if not reusable:
            marker.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            OmegaConf.save(OmegaConf.create(plan["config"]), point / "config.yaml")
            command = [
                sys.executable,
                str(Path(evaluation_entrypoint).resolve()),
                "--config-path",
                str(point),
                "--config-name",
                "config",
                f"hydra.run.dir={point}",
            ]
            print(
                f"[{index + 1}/{len(plans)}] {plan['axes']}, seed={plan['seed']}; log: {point / 'evaluation.log'}",
                flush=True,
            )
            with (point / "evaluation.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            if not result_path.exists():
                raise RuntimeError(f"evaluation did not produce {result_path}")
        result = json.loads(result_path.read_text())
        row = {
            **dict(scalar_metrics(result)),
            **plan["axes"],
            "point_id": plan["point_id"],
            "seed": plan["seed"],
            "result_path": str(result_path),
        }
        # Missing/nonfinite objectives fail explicitly, never silently improve a frontier.
        pareto_flags([row], objectives, options["group_by"])
        write_json(marker, {"signature": plan["signature"]})
        rows.append(row)
        summarize(output, rows, objectives, options["group_by"])
    print(f"Completed {len(rows)} evaluations: {output / 'frontiers.csv'}", flush=True)
    return rows
