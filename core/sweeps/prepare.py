"""Export Hydra multiruns for the existing queue; never launch training.

python -m core.sweeps.prepare -m +sweep=gaussian preparation.output=/tmp/plan
python -m core.sweeps.prepare collect /tmp/plan
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .configuration import capture_runtime_paths
from .nested_tuning import validate


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
    if metadata.get("reference_manifest"):
        references = json.loads(
            Path(metadata["reference_manifest"]).expanduser().read_text()
        )
        reference = next(
            c for c in references if c["name"] == metadata["reference_case"]
        )
        evaluation.update(reference["evaluation"])
    evaluation.update(metadata["evaluation"])
    config = capture_runtime_paths(config)
    del config["preparation"]
    sampler = instantiate(config.sampler)
    config.sampler.steps = getattr(sampler, "steps_for_nfe", lambda n: n - 1)(
        evaluation["nfe"]
    )
    recipe = {
        key: OmegaConf.to_container(config[key], resolve=False)
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
    config.wandb = {
        **metadata["wandb"],
        "group": metadata["name"],
        "name": name,
        "job_type": "train",
    }
    config.output_dir = str(output / "runs" / name)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
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
        wandb=OmegaConf.to_container(config.wandb),
        resources=metadata["local"]["resources_per_job"],
    )
    validate(case)
    destination.mkdir(parents=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
    (destination / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    (destination / "job.json").write_text(
        json.dumps(
            dict(name=name, config=resolved, checkpoint="last.ckpt", inference=[])
        )
        + "\n"
    )
    return name


def collect(output):
    """Collect exported cases into the existing queue and nested-tuning formats."""
    output = Path(output).expanduser().resolve()
    cases, jobs = [], []
    for directory in sorted((output / "cases").iterdir()):
        case = json.loads((directory / "case.json").read_text())
        job = json.loads((directory / "job.json").read_text())
        validate(case)
        if case["name"] != job["name"]:
            raise ValueError(f"Mismatched case and job in {directory}")
        cases.append(case)
        jobs.append(job)
    if not cases:
        raise ValueError("No prepared cases")
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case names")
    (output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    OmegaConf.save(OmegaConf.create(jobs), output / "training-jobs.yaml")
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
