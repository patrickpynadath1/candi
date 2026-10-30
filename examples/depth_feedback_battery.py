"""Double pure continuous backbones and compare four feedback strategies."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from core.sweeps.nested_tuning import validate


def build(manifest, output, project):
    cases = []
    for original in json.loads(Path(manifest).read_text()):
        if not any(
            f"-{family}-" in original["name"]
            and original["name"].endswith("-sc-on_20m")
            for family in ("gaussian", "simplex", "spherical")
        ):
            continue
        base = OmegaConf.load(original["config"])
        for variant in ("none", "posterior_mean", "hidden", "token_feedback"):
            config = copy.deepcopy(base)
            config.model.backbone.n_blocks *= 2
            depth = config.model.backbone.n_blocks
            name = original["name"].split("-sc-on_20m")[0] + f"-depth-{depth}-{variant}"
            config.self_conditioning.enabled = variant in ("posterior_mean", "hidden")
            config.self_conditioning.source = (
                variant if config.self_conditioning.enabled else "posterior_mean"
            )
            config.token_feedback = dict(
                enabled=variant == "token_feedback",
                depth=depth // 2 - 1,
                loss_weight=1.0,
                detach=False,
            )
            config.checkpoint.resume = None
            config.evaluation.checkpoint = None
            config.wandb.update(
                dict(project=project, group=project, name=name, job_type="train")
            )
            recipe = {
                key: OmegaConf.to_container(config[key], resolve=False)
                for key in ("model", "self_conditioning", "token_feedback", "objective")
            }
            config.algorithm_definition = recipe
            config.benchmark.algorithm_name = name
            config.benchmark.algorithm_id = hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()
            ).hexdigest()
            config.benchmark.battery_name = project
            path = Path(output) / "configs" / f"{name}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, path)
            spec = copy.deepcopy(original)
            spec.update(
                name=name,
                config=str(path.resolve()),
                training=dict(space={}, trials=1),
                wandb=OmegaConf.to_container(config.wandb),
            )
            validate(spec)
            cases.append(spec)
    if len(cases) != 36 or len({c["name"] for c in cases}) != 36:
        raise ValueError(f"Expected 36 distinct cases; got {len(cases)}")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    cases = build(args.manifest, args.output, args.project)
    (args.output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Prepared {len(cases)} cases")
