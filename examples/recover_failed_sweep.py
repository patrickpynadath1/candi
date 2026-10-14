"""Requeue failed runs through the live queue, retaining available training state."""

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.sweeps.jobs import TrainingJob, InferenceJob
from core.sweeps.submissions import SubmissionQueue


SUFFIX = "/repair-v1"


def recover(experiment, queue):
    snapshots = list(experiment.glob("experiment_state*.json"))
    if not snapshots:
        print("Waiting for Ray experiment state", flush=True)
        return False
    try:
        state = json.loads(max(snapshots, key=lambda p: p.stat().st_mtime).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        print("Waiting for Ray to finish publishing experiment state", flush=True)
        return False
    submissions, _ = queue.read_after(0)
    originals = {
        job["name"]
        for _, job in submissions
        if not job["trial"]["config"].get("benchmark", {}).get("recovery_of")
    }
    accepted = {job["name"] for _, job in submissions}
    terminal = set()
    jobs = []
    for record, _ in state["trial_data"]:
        trial = json.loads(record)
        spec = trial["config"]
        if spec["stage"] != "train" or spec["name"] not in originals:
            continue
        if trial["status"] in {"TERMINATED", "ERROR"}:
            terminal.add(spec["name"])
        if trial["status"] != "ERROR":
            continue
        if spec["name"] + SUFFIX in accepted:
            continue
        directory = next(experiment.glob(f"*{trial['trial_id']}*"), None)
        try:
            if directory is None:
                raise FileNotFoundError(trial["trial_id"])
            error = (directory / "error.txt").read_text()
        except FileNotFoundError:
            terminal.discard(spec["name"])
            print(f"Waiting for error artifacts: {spec['name']}", flush=True)
            continue
        if not error.strip():
            terminal.discard(spec["name"])
            continue
        reporting_error = "could not convert string to float:" in error and any(
            x in error for x in ("'True'", "'False'")
        )
        blur_error = (
            "core/modifiers/self_conditioning/prediction.py" in error
            and "IndexError: The shape of the mask" in error
        )
        if not (reporting_error or blur_error):
            raise RuntimeError(
                f"Unrecognized failure: {spec['name']}; inspect {directory / 'error.txt'}"
            )
        checkpoint = directory / "run/checkpoints/last.ckpt"
        if reporting_error and not checkpoint.is_file():
            terminal.discard(spec["name"])
            print(f"Waiting for checkpoint publication: {checkpoint}", flush=True)
            continue
        config = copy.deepcopy(spec["trial"]["config"])
        config["checkpoint"]["resume"] = (
            str(checkpoint.resolve()) if checkpoint.is_file() else None
        )
        config["benchmark"]["recovery_of"] = trial["trial_id"]
        if config.get("wandb"):
            config["wandb"]["name"] += SUFFIX
        inference = tuple(
            replace(
                InferenceJob(**choice),
                overrides={
                    **choice["overrides"],
                    **(
                        {"wandb.name": choice["overrides"]["wandb.name"] + SUFFIX}
                        if "wandb.name" in choice["overrides"]
                        else {}
                    ),
                },
            )
            for choice in spec["inference"]
        )
        jobs.append(
            TrainingJob(
                name=spec["name"] + SUFFIX,
                config=config,
                inference=inference,
                checkpoint=spec["checkpoint"],
            )
        )
    added = queue.submit(jobs) if jobs else 0
    print(
        f"Accepted {added} recovery jobs; {len(terminal)}/{len(originals)} original training jobs terminal",
        flush=True,
    )
    return terminal == originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Also recover matching failures from original workers still running; exit once all originals finish",
    )
    args = parser.parse_args()
    queue = SubmissionQueue(args.queue)
    while True:
        finished = recover(args.experiment, queue)
        if finished or not args.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
