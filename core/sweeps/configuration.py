"""Compose method configurations for Python callers, including Ray Tune."""

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
