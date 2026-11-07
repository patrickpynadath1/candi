"""Compose method configurations and prepare per-run configs for Ray workers."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

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
    return capture_runtime_paths(config)


def capture_runtime_paths(config: DictConfig) -> DictConfig:
    """Capture Hydra paths for deferred jobs, preserving recipe interpolations."""

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


def run_configuration(spec):
    """Accept either a Hydra preset specification or an editable config snapshot."""
    unknown = set(spec) - {"experiment", "overrides", "config"}
    if unknown:
        raise ValueError(f"Unknown run options: {sorted(unknown)}")
    if "config" in spec:
        if "experiment" in spec or "overrides" in spec:
            raise ValueError("Use either config or experiment/overrides, not both")
        return OmegaConf.create(spec["config"])
    return compose(spec.get("experiment"), spec.get("overrides", {}))


def prepare_run(spec, output, resources):
    """Resolve one worker's config for its Ray reservation and output directory.

    Workers hold zero or one whole GPU; the reservation sets the device, so a
    run never uses a GPU it was not given.
    """
    config = run_configuration(spec)
    gpus = resources.get("GPU", 0)
    if isinstance(gpus, bool) or not isinstance(gpus, (int, float)) or gpus not in (0, 1):
        raise ValueError("a run requires zero or one whole GPU")
    gpus = int(gpus)
    if config.trainer.get("num_nodes", 1) != 1:
        raise ValueError("a run supports one node")
    if config.trainer.strategy not in ("auto", None) or config.get("strategy") is not None:
        raise ValueError("Zero/one-GPU runs require strategy=auto and no strategy config")
    config.trainer.devices = 1
    config.output_dir = str(Path(output).resolve())
    config.trainer.accelerator = "gpu" if gpus else "cpu"
    config.evaluation.device = "cuda:0" if gpus else "cpu"
    # Freeze interpolations only after assigning the per-run output path.
    return OmegaConf.create(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )
