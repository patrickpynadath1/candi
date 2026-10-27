"""Submit a strength sweep of the existing Alpha procedure on masked bases."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps import InferenceJob
from core.sweeps.submissions import SubmissionQueue
from examples.distillation_effectiveness import build_comparison
from examples.distillation_smoke import discover


def build_jobs(bases):
    jobs = []
    for template in build_comparison(bases):
        if not template.name.startswith("fewstep-3000-alpha-"):
            continue
        for k in (4, 8, 16):
            config = OmegaConf.create(
                OmegaConf.to_container(template.config, resolve=False)
            )
            config.procedure.phases[0].objective.k = k
            name = template.name.replace("-alpha-", f"-alpha-k{k}-")
            config.wandb.name = name + "/train"
            config.wandb.group = "alpha-strength-3000"
            inference = tuple(
                InferenceJob(
                    overrides={
                        **i.overrides,
                        "wandb.name": name + "/" + i.overrides["evaluation.policy"],
                    }
                )
                for i in template.inference
            )
            jobs.append(
                replace(template, name=name, config=config, inference=inference)
            )
    return jobs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument(
        "--base-root",
        type=Path,
        default=Path.home() / ".cache/duo/batteries/three-tasks-20m-bf16-20261012",
    )
    args = p.parse_args()
    jobs = build_jobs(discover(args.base_root))
    rows = []
    for j in jobs:
        row = asdict(j)
        row["config"] = OmegaConf.to_container(j.config, resolve=True)
        rows.append(row)
    args.plan.write_text(json.dumps(rows, indent=2) + "\n")
    print("Submitted", SubmissionQueue(args.queue).submit(jobs), "Alpha jobs")
