"""Queue posterior-mean SC counterparts of existing pure and hybrid recipes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from core.sweeps.nested_tuning import validate


def build(manifests, output, project):
    output = Path(output)
    cases = []
    for manifest in manifests:
        for original in json.loads(Path(manifest).read_text()):
            config = OmegaConf.load(original["config"])
            if not config.self_conditioning.enabled or "insert" in original["name"]:
                continue
            if config.self_conditioning.source != "hidden":
                raise ValueError("Expected hidden-SC source recipes")
            spec = copy.deepcopy(original)
            name = original["name"] + "-posterior-mean"
            config.self_conditioning.source = "posterior_mean"
            config.checkpoint.resume = None
            config.evaluation.checkpoint = None
            config.wandb.update(
                dict(project=project, group=project, name=name, job_type="train")
            )
            recipe = dict(
                model=OmegaConf.to_container(config.model, resolve=False),
                self_conditioning=OmegaConf.to_container(config.self_conditioning),
                objective=OmegaConf.to_container(config.objective),
            )
            config.algorithm_definition = recipe
            config.benchmark.algorithm_name = (
                str(config.benchmark.algorithm_name) + "-posterior-mean"
            )
            config.benchmark.algorithm_id = hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()
            ).hexdigest()
            config.benchmark.battery_name = project
            path = output / "configs" / f"{name}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, path)
            spec.update(
                name=name,
                config=str(path.resolve()),
                training=dict(space={}, trials=1),
                wandb=OmegaConf.to_container(config.wandb),
            )
            validate(spec)
            cases.append(spec)
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Duplicate cases across source manifests")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    cases = build(args.manifests, args.output, args.project)
    (args.output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Prepared {len(cases)} posterior-mean SC cases")
