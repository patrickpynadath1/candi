"""Continuous/discrete hybrids crossed with self-conditioning and adaptive clocks."""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

from omegaconf import OmegaConf
from core.sweeps.configuration import ROOT
from core.sweeps.nested_tuning import validate


def build(previous, output, project):
    output = Path(output)
    cases = []
    for task, family, discrete, sc, adaptive in itertools.product(
        ["sudoku_hard", "text8", "tinystories"],
        ["gaussian", "spherical", "simplex"],
        ["uniform", "masked"],
        [False, True],
        [False, True],
    ):
        original = next(
            c
            for c in previous
            if c["name"].startswith(f"{task}-{family}-")
            and c["name"].endswith("sc-off_20m")
        )
        config = OmegaConf.load(original["config"])
        view = family if family == "gaussian" else family + "_rank"
        continuous = OmegaConf.load(ROOT / f"configs/hybrid_view/{view}.yaml")
        if adaptive:
            continuous.time_map = OmegaConf.load(ROOT / "configs/time_map/cdcd.yaml")
        config.model.views = dict(
            content=continuous,
            discrete=OmegaConf.load(ROOT / f"configs/hybrid_view/{discrete}.yaml"),
        )
        config.self_conditioning.enabled = sc
        name = f"{family}-{discrete}-sc-{int(sc)}-" + (
            "adaptive" if adaptive else "rank-linear"
        )
        case_name = f"{task}-{name}"
        config.checkpoint.resume = None
        config.evaluation.checkpoint = None
        config.wandb.update(
            dict(project=project, group=project, name=case_name, job_type="train")
        )
        # Record the composed recipe rather than retaining the source single-view identity.
        recipe = dict(
            model=OmegaConf.to_container(config.model, resolve=False),
            self_conditioning=OmegaConf.to_container(config.self_conditioning),
            objective=OmegaConf.to_container(config.objective),
        )
        config.algorithm_definition = recipe
        config.benchmark.algorithm_name = name
        config.benchmark.algorithm_id = hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode()
        ).hexdigest()
        config.benchmark.battery_name = project
        config.benchmark.schedule_adaptation = "cdcd" if adaptive else "none"
        path = output / "configs" / f"{case_name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, path)
        case = copy.deepcopy(original)
        case.update(
            name=case_name,
            config=str(path.resolve()),
            training=dict(space={}, trials=1),
            wandb=OmegaConf.to_container(config.wandb),
        )
        validate(case)
        cases.append(case)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    cases = build(
        json.loads(args.previous_manifest.read_text()), args.output, args.project
    )
    (args.output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Prepared {len(cases)} training + inference cases")
