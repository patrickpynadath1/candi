"""Regenerate nested-tuning final test evaluations and compare them exactly.

For each case directory, rebuild the final evaluation from its own records
(final/result.json selection + protocol.json test partition and seeds), run it
with the current code into a fresh output directory, and compare every sample
(index, generated text, NFE) and the final score with the stored results.
Never trains, never writes into the case directory, never logs to W&B.

  python scripts/analysis/regenerate_final.py OUTPUT CASE_DIR [CASE_DIR ...]
"""

import argparse
import json
from pathlib import Path
import sys
import traceback


def final_payload(case):
    selection = json.loads((case / "final/result.json").read_text())["selection"]
    evaluation = json.loads((case / "protocol.json").read_text())["spec"]["evaluation"]
    # The same construction nested_tuning uses for its final evaluation.
    hidden = ("test_indices", "test_seeds", "test_dataset", "tuning_indices")
    validation = {k: v for k, v in evaluation.items() if k not in hidden}
    validation["indices"] = evaluation["tuning_indices"]
    final = {
        **validation,
        "dataset": evaluation.get("test_dataset", validation.get("dataset", "validation")),
        "indices": evaluation["test_indices"],
        "seeds": evaluation["test_seeds"],
    }
    return {
        "checkpoint": selection["checkpoint"],
        "policy": selection["policy"],
        "parameters": selection["inference_parameters"],
        "evaluation": final,
        "resources": {"cpu": 4, "gpu": 1},
    }


def records(path):
    data = json.loads(path.read_text())
    return data, [(r["index"], r.get("generated"), r.get("nfe")) for r in data["records"]]


def compare(case, output, seeds):
    report = {"seeds": {}}
    for seed in seeds:
        old, old_records = records(case / f"final/seed-{seed}/results.json")
        new, new_records = records(output / f"seed-{seed}/results.json")
        differing = [i for i, (a, b) in enumerate(zip(old_records, new_records)) if a != b]
        report["seeds"][seed] = {
            "samples": len(old_records),
            "identical_samples": len(old_records) - len(differing)
            if len(old_records) == len(new_records)
            else 0,
            "same_count": len(old_records) == len(new_records),
            "first_difference": differing[:1],
        }
    report["identical"] = all(
        s["same_count"] and s["identical_samples"] == s["samples"]
        for s in report["seeds"].values()
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="+", type=Path)
    args = parser.parse_args()

    import core.training.runner as runner
    from core.sweeps import nested_tuning

    original_run = runner.run

    def run_without_tracking(config):
        config.wandb = None
        return original_run(config)

    runner.run = run_without_tracking
    summary = []
    for case in args.cases:
        case = case.resolve()
        output = args.output.resolve() / case.parent.name / case.name
        if output.exists():
            raise FileExistsError(output)
        entry = {"case": str(case), "output": str(output)}
        try:
            payload = final_payload(case)
            entry["sampler"] = payload["policy"]["name"]
            score = nested_tuning._evaluate_worker(payload, output)["score"]
            stored = json.loads((case / "final/result.json").read_text())["score"]
            entry.update(score=score, stored_score=stored, score_matches=score == stored)
            entry.update(compare(case, output, payload["evaluation"]["seeds"]))
        except Exception:
            entry["error"] = traceback.format_exc()
        summary.append(entry)
        print(json.dumps(entry, indent=2), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    identical = sum(bool(e.get("identical") and e.get("score_matches")) for e in summary)
    print(f"{identical}/{len(summary)} cases reproduced exactly", flush=True)
    sys.exit(0 if identical == len(summary) else 1)


if __name__ == "__main__":
    main()
