"""Gaussian doubled-depth gradient-routing ablation with identity SC injection."""

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
        if "-gaussian-" not in original["name"] or not original["name"].endswith(
            "-posterior_mean"
        ):
            continue
        for mode in ("both", "producer", "mixed"):
            config = OmegaConf.load(original["config"])
            name = original["name"] + "-grad-" + mode
            config.model.conditioning_initialization = "identity"
            config.self_conditioning.gradient_mode = mode
            config.prediction.gradient_mode = "${self_conditioning.gradient_mode}"
            config.wandb.update(
                dict(project=project, group=project, name=name, job_type="train")
            )
            config.algorithm_definition = {
                key: OmegaConf.to_container(config[key], resolve=False)
                for key in ("model", "self_conditioning", "token_feedback", "objective")
            }
            config.benchmark.algorithm_name = name
            config.benchmark.algorithm_id = hashlib.sha256(
                json.dumps(
                    OmegaConf.to_container(config.algorithm_definition), sort_keys=True
                ).encode()
            ).hexdigest()
            config.benchmark.battery_name = project
            path = Path(output) / "configs" / f"{name}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, path)
            case = copy.deepcopy(original)
            case.update(
                name=name,
                config=str(path.resolve()),
                wandb=OmegaConf.to_container(config.wandb),
            )
            validate(case)
            cases.append(case)
    if len(cases) != 9:
        raise ValueError(f"Expected 9 cases; got {len(cases)}")
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
