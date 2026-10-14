"""Build fresh training + tuning studies using an existing evaluation protocol."""

import argparse
import copy
import json
from pathlib import Path

from omegaconf import OmegaConf

from core.sweeps.nested_tuning import validate
from examples.self_conditioning_battery import build_jobs
from examples.insertion_head_sweep import build_jobs as insertion_jobs


def build(previous, output, project, *, adaptive_continuous=False):
    output = Path(output)
    tasks = sorted({s["name"].split("-")[0] for s in previous})
    jobs = build_jobs(
        output.name,
        tasks=tasks,
        project=project,
        adaptive_continuous=adaptive_continuous,
    )
    jobs.extend(insertion_jobs(output.name, tasks=tasks, project=project))
    configs = {
        f"{j.config['benchmark']['task_id']}-{j.config['benchmark']['algorithm_name']}": j.config
        for j in jobs
    }
    studies = []
    for original in previous:
        spec = copy.deepcopy(original)
        config = OmegaConf.create(configs[spec["name"]])
        config.checkpoint.resume = None
        config.evaluation.checkpoint = None
        config.wandb.group = output.name
        config.wandb.name = spec["name"]
        config.benchmark.schedule_adaptation = (
            "cdcd"
            if OmegaConf.select(config, "model.views.content.time_map")
            else "none"
        )
        path = output / "configs" / f"{spec['name']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, path)
        spec["config"] = str(path.resolve())
        spec["training"] = {"space": {}, "trials": 1}
        spec["wandb"] = OmegaConf.to_container(config.wandb, resolve=True)
        validate(spec)
        studies.append(spec)
    return studies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--adaptive-continuous", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    studies = build(
        json.loads(args.previous_manifest.read_text()),
        args.output,
        args.project,
        adaptive_continuous=args.adaptive_continuous,
    )
    (args.output / "manifest.json").write_text(json.dumps(studies, indent=2))
    print(f"Prepared {len(studies)} fresh training + tuning studies in {args.output}")
