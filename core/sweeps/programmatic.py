"""Compose independent method configurations and queue isolated local runs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from hydra import compose as hydra_compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def _hydra_literal(value):
    if isinstance(value, Mapping):
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)) for key in value):
            raise ValueError(
                "Mapping override keys must be identifiers; use an editable config for other keys"
            )
        return (
            "{"
            + ",".join(f"{key}:{_hydra_literal(item)}" for key, item in value.items())
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_literal(item) for item in value) + "]"
    return json.dumps(value)


def compose(
    experiment: str | None = None,
    overrides: Mapping[str, Any] | Sequence[str] = (),
    *,
    config_dir: str | Path | None = None,
) -> DictConfig:
    """Compose a preset and Hydra group/field overrides, preserving interpolation.

    Strings in mappings are literal strings; use a list of Hydra overrides for
    advanced grammar (+, ~, package overrides). Method presets use +experiment.
    Composition must be called outside an active Hydra initialization context.
    """
    directory = Path(config_dir).resolve() if config_dir else ROOT / "configs"
    arguments = [f"+experiment={experiment}"] if experiment else []
    if isinstance(overrides, Mapping):
        arguments.extend(
            f"{key}={_hydra_literal(value)}" for key, value in overrides.items()
        )
    else:
        arguments.extend(overrides)
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = hydra_compose(config_name="config", overrides=arguments)

    # Capture only Hydra's runtime paths. Keep task/model interpolations live and
    # do not replace Hydra's global resolver (which would affect other callers).
    def capture(value):
        if isinstance(value, str):
            return value.replace("${hydra:runtime.cwd}", str(ROOT)).replace(
                "${hydra:runtime.output_dir}", "${output_dir}"
            )
        if isinstance(value, dict):
            return {key: capture(item) for key, item in value.items()}
        if isinstance(value, list):
            return [capture(item) for item in value]
        return value

    values = capture(OmegaConf.to_container(config, resolve=False))
    if values["output_dir"] == "${output_dir}":
        values["output_dir"] = str(ROOT / "core/outputs/programmatic")
    return OmegaConf.create(values)


@dataclass
class Trial:
    experiment: str | None = None
    overrides: Mapping[str, Any] | Sequence[str] = field(default_factory=dict)
    config: DictConfig | None = None
    name: str | None = None

    def compose(self) -> DictConfig:
        if self.config is not None:
            if self.experiment is not None or self.overrides:
                raise ValueError("Use either config or experiment/overrides, not both")
            return copy.deepcopy(self.config)
        return compose(self.experiment, self.overrides)


def grid(
    experiments: Iterable[str],
    axes: Mapping[str, Sequence[Any]] | None = None,
    *,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> list[Trial]:
    """Cartesian expansion; use explicit Trial lists for conditional choices."""
    axes = dict(axes or {})
    if seeds is not None:
        if "seed" in axes:
            raise ValueError("Specify seed either in axes or seeds")
        axes["seed"] = seeds
    return [
        Trial(
            experiment, copy.deepcopy(dict(overrides or {}) | dict(zip(axes, values)))
        )
        for experiment in experiments
        for values in itertools.product(*axes.values())
    ]


@dataclass(frozen=True)
class Resources:
    gpus: int = 1
    cpus: int = 4
    gpu_ids: Sequence[str | int] | None = None
    max_parallel: int | None = None

    def __post_init__(self):
        for name in ("gpus", "cpus", "max_parallel"):
            value = getattr(self, name)
            if value is None and name == "max_parallel":
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.gpus not in (0, 1):
            raise ValueError(
                "Only zero or one GPU per trial is supported; multi-GPU trials are not supported"
            )
        if self.cpus < 1 or (self.max_parallel is not None and self.max_parallel < 1):
            raise ValueError("cpus and max_parallel must be positive")
        if self.gpus == 0 and self.gpu_ids is not None:
            raise ValueError("gpu_ids requires gpus=1")

    def slots(self) -> list[str | None]:
        if self.gpus not in (0, 1):
            raise ValueError(
                "Only zero or one GPU per trial is supported; multi-GPU trials are not supported"
            )
        if self.cpus < 1 or (self.max_parallel is not None and self.max_parallel < 1):
            raise ValueError("cpus and max_parallel must be positive")
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        if self.cpus > cpu_count:
            raise ValueError(
                f"Requested {self.cpus} CPUs per trial, only {cpu_count} available"
            )
        limit = min(cpu_count // self.cpus, self.max_parallel or cpu_count)
        if self.gpus == 0:
            if self.gpu_ids is not None:
                raise ValueError("gpu_ids requires gpus=1")
            return [None] * limit
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = (
            [
                item.strip()
                for item in visible.split(",")
                if item.strip() and item.strip() != "-1"
            ]
            if visible is not None
            else None
        )
        if self.gpu_ids is not None:
            devices = [str(item) for item in self.gpu_ids]
            if allowed is not None and not set(devices).issubset(allowed):
                raise ValueError(
                    "gpu_ids must be a subset of CUDA_VISIBLE_DEVICES (physical IDs or UUIDs)"
                )
        elif allowed is not None:
            devices = allowed
        else:
            import torch

            devices = [str(index) for index in range(torch.cuda.device_count())]
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("No GPUs available, or duplicate GPU IDs requested")
        return devices[:limit]


@dataclass
class TrialResult:
    name: str
    status: str
    output_dir: str
    returncode: int | None = None
    metrics: dict = field(default_factory=dict)


class SweepFailed(RuntimeError):
    def __init__(self, results):
        self.results = results
        super().__init__(
            "Failed trials: "
            + ", ".join(row.name for row in results if row.status != "succeeded")
            + "; see each run.log"
        )


class Sweep:
    def __init__(
        self,
        trials: Iterable[Trial | DictConfig],
        *,
        output_dir: str | Path,
        resources: Resources | None = None,
    ):
        self.trials = [
            trial if isinstance(trial, Trial) else Trial(config=trial)
            for trial in trials
        ]
        self.output_dir = Path(output_dir).resolve()
        self.resources = resources or Resources()

    def plan(self) -> list[dict]:
        """Resolve configs without launching, creating files or probing GPUs."""
        plans, names = [], set()
        for index, trial in enumerate(self.trials):
            name = trial.name or f"{index:04d}-{trial.experiment or 'trial'}"
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
                raise ValueError(f"Invalid or duplicate trial name: {name!r}")
            names.add(name)
            config = trial.compose()
            if (
                int(config.trainer.devices) != 1
                or int(config.trainer.get("num_nodes", 1)) != 1
            ):
                raise ValueError(
                    "Sweep trials must use trainer.devices=1 and num_nodes=1"
                )
            if (
                config.trainer.strategy not in ("auto", None)
                or config.get("strategy") is not None
            ):
                raise ValueError("Sweep trials require the single-device auto strategy")
            config.output_dir = str(self.output_dir / name)
            config.trainer.accelerator = "gpu" if self.resources.gpus else "cpu"
            config.evaluation.device = "cuda:0" if self.resources.gpus else "cpu"
            values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            fingerprint = hashlib.sha256(
                json.dumps(values, sort_keys=True).encode()
            ).hexdigest()
            plans.append(
                dict(
                    name=name,
                    config=values,
                    fingerprint=fingerprint,
                    output_dir=config.output_dir,
                )
            )
        return plans

    def run(
        self, *, resume: bool = True, dry_run: bool = False
    ) -> list[TrialResult] | list[dict]:
        plans = self.plan()
        if dry_run:
            return plans
        from core.sweeps.local import LocalExecutor

        rows = LocalExecutor(self.output_dir, self.resources).run(plans, resume=resume)
        results = [
            TrialResult(
                **{
                    key: row[key]
                    for key in ("name", "status", "output_dir", "returncode", "metrics")
                }
            )
            for row in rows
        ]
        if any(row.status != "succeeded" for row in results):
            raise SweepFailed(results)
        return results


def run_sweep(
    trials: Iterable[Trial | DictConfig],
    *,
    output_dir: str | Path,
    resources: Resources | None = None,
    resume: bool = True,
    dry_run: bool = False,
):
    return Sweep(trials, output_dir=output_dir, resources=resources).run(
        resume=resume, dry_run=dry_run
    )
