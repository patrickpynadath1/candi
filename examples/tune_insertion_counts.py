"""Build fixed-checkpoint insertion tuning cases from a completed experiment."""

import argparse
import copy
import json
from pathlib import Path


def build_cases(source, project, trials=24):
    source = Path(source)
    cases = []
    for original in json.loads((source / "manifest.json").read_text()):
        policies = original["inference"]
        if not all(
            p["sampler"]["_target_"].endswith(".InsertionSampler") for p in policies
        ):
            continue
        result = json.loads(
            (source / original["name"] / "final/result.json").read_text()
        )
        checkpoint = result["selection"]["checkpoint"]
        case = copy.deepcopy(original)
        case["config"] = checkpoint["config"]
        case["training"] = dict(space={}, trials=1, checkpoint=checkpoint["checkpoint"])
        case["wandb"].update(
            project=project, group=project, job_type="inference-tuning"
        )
        winner = result["selection"]
        policy = copy.deepcopy(winner["policy"])
        case["inference"] = [policy]
        # Keep the selected token temperature and decoding policy fixed.
        policy["sampler"]["count_scale"] = 1.0
        policy["space"] = {
            "sampler.count_scale": dict(type="float", low=0.25, high=4.0, log=True)
        }
        policy["space"].update(
            {key: [value] for key, value in winner["inference_parameters"].items()}
        )
        policy["trials"] = trials
        policy["initial_points"] = [
            {**winner["inference_parameters"], "sampler.count_scale": scale}
            for scale in [1.0, 0.5, 2.0]
        ]
        from core.sweeps.nested_tuning import validate

        validate(case)
        cases.append(case)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()
    cases = build_cases(args.source, args.project, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Wrote {len(cases)} fixed-checkpoint cases to {args.output}")
