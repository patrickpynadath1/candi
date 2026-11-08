"""Export Hydra multiruns as a nested-tuning manifest; never launch training.

python -m core.sweeps.prepare -m +sweep=gaussian preparation.output=$SDDC_STORAGE_ROOT/plans/X
python -m core.sweeps.prepare collect $SDDC_STORAGE_ROOT/plans/X
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .budget import sampler_steps
from .configuration import capture_runtime_paths
from .nested_tuning import validate


def load_evaluation(path, task_name):
    """A committed evaluation spec for ``task_name`` (repo-relative or absolute path).

    The JSON holds ``task`` and the evaluation fields: tuning/test indices, score,
    direction, seeds and task overrides; ``source`` records where it came from.
    """
    path = Path(path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[2] / path
    spec = json.loads(path.read_text())
    # Indices only mean something for the task they were drawn from.
    if spec.get("task") != task_name:
        raise ValueError(
            f"evaluation file {path} is for task {spec.get('task')!r}, "
            f"not {task_name!r}; use that task's file"
        )
    return {key: value for key, value in spec.items() if key not in ("task", "source")}


def prepare_case(config):
    """Serialize one already-composed Hydra job and its tuning specification."""
    metadata = OmegaConf.to_container(config.preparation, resolve=True)
    name = metadata["case_name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid job name {name!r}")
    output = Path(metadata["output"]).expanduser().resolve()
    destination = output / "cases" / name
    path = output / "configs" / f"{name}.yaml"
    if destination.exists():
        raise FileExistsError(destination)
    if path.exists():
        raise FileExistsError(path)
    evaluation = {}
    if metadata.get("evaluation_file"):
        evaluation.update(load_evaluation(metadata["evaluation_file"], metadata["task_name"]))
    evaluation.update(metadata["evaluation"])
    config = capture_runtime_paths(config)
    del config["preparation"]
    sampler = instantiate(config.sampler)
    budget_parameter = metadata.get("sampler_budget_parameter", "steps")
    if budget_parameter is not None:
        config.sampler[budget_parameter] = sampler_steps(
            config, sampler, evaluation["nfe"]
        )
    recipe = {
        key: None if config[key] is None else OmegaConf.to_container(config[key], resolve=False)
        for key in metadata["recipe_fields"]
    }
    config.algorithm_definition = recipe
    config.benchmark = dict(
        algorithm_id=hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode()
        ).hexdigest(),
        algorithm_name=metadata["recipe_name"],
        battery_name=metadata["name"],
        task_id=metadata["task_name"],
        training_seed=config.seed,
    )
    # Tracking only when a W&B entity is configured (WANDB_ENTITY).
    config.wandb = (
        None
        if metadata["wandb"].get("entity") is None
        else {**metadata["wandb"], "group": metadata["name"], "name": name, "job_type": "train"}
    )
    config.output_dir = str(output / "runs" / name)
    # Fail at export, not on a worker, if anything is unresolved.
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    policies = []
    for policy in metadata["inference"]:
        native = policy.get("sampler") or OmegaConf.to_container(
            config.sampler, resolve=True
        )
        policy["sampler"] = {**native, **policy.pop("sampler_overrides", {})}
        policies.append(policy)
    case = dict(
        name=name,
        config=str(path),
        training=metadata["training"],
        evaluation=evaluation,
        inference=policies,
        wandb=None if config.wandb is None else OmegaConf.to_container(config.wandb),
        resources=metadata["local"]["resources_per_job"],
    )
    validate(case)
    destination.mkdir(parents=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
    (destination / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return name


def collect(output):
    """Collect exported cases into the nested-tuning manifest."""
    output = Path(output).expanduser().resolve()
    cases = []
    for directory in sorted((output / "cases").iterdir()):
        case = json.loads((directory / "case.json").read_text())
        validate(case)
        cases.append(case)
    if not cases:
        raise ValueError("No prepared cases")
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case names")
    (output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    return len(cases)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config):
    print(f"Prepared {prepare_case(config)}; no jobs submitted.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        parser = argparse.ArgumentParser(description=collect.__doc__)
        parser.add_argument("output", type=Path)
        args = parser.parse_args(sys.argv[2:])
        print(f"Collected {collect(args.output)} cases; no jobs submitted.")
    else:
        main()
