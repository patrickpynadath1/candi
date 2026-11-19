"""Sudoku hint ladder: distinct valid solutions as clues are removed, per recipe and sampler.

Each job evaluates one Part 2 Sudoku checkpoint with one sampler (a tuned Part 2 policy
winner, or the native sampler at its untuned starting point) on task=sudoku_diversity at
one retained-clue fraction. No tuning happens here.

    python sudoku_hint_ladder.py jobs SPEC.json OUT_DIR       # print one job id per line
    python sudoku_hint_ladder.py run SPEC.json OUT_DIR JOB_ID  # run one job (uses the visible GPU)
    python sudoku_hint_ladder.py summary OUT_DIR               # table of all finished jobs

SPEC.json (built from the Part 2 results): a list of cases with checkpoint, config, nfe and
samplers [{name, rank, val, policy, parameters}].
"""

import json
import sys
from pathlib import Path

FRACTIONS = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
REPETITIONS = 16
NUM_PUZZLES = 32
SEED = 0  # sampling seed, the same for every job
HINT_SEED = 0  # clue-removal seed (puzzle i uses HINT_SEED + i)


def job_ids(spec):
    return [f"{c['case']}|{s['name']}|{f}" for c in spec for s in c["samplers"] for f in FRACTIONS]


def run_job(spec, out_dir, job):
    from omegaconf import OmegaConf

    from core.sweeps.configuration import prepare_run
    from core.sweeps.nested_tuning import evaluation_config
    from core.training.runner import run

    case_name, sampler_name, fraction = job.split("|")
    case = next(c for c in spec if c["case"] == case_name)
    sampler = next(s for s in case["samplers"] if s["name"] == sampler_name)
    destination = Path(out_dir) / case_name / sampler_name / f"f{fraction}"
    if (destination / "results.json").exists():
        return
    config = evaluation_config(OmegaConf.load(case["config"]), {"nfe": case["nfe"]},
                               sampler["policy"], sampler["parameters"])
    task = OmegaConf.to_container(config.task, resolve=False)
    assert task["_target_"] == "core.tasks.sudoku.SudokuTask", task["_target_"]
    task.update(_target_="core.tasks.sudoku_diversity.SudokuDiversityTask",
                retained_fraction=float(fraction), repetitions=REPETITIONS,
                num_puzzles=NUM_PUZZLES, seed=HINT_SEED)
    config.task = OmegaConf.create(task)
    config.mode = "evaluate"
    config.compile = False
    config.seed = SEED
    config.evaluation.checkpoint = case["checkpoint"]
    config.evaluation.legacy = False
    config.evaluation.max_examples = None
    config.evaluator = {"_target_": "core.evaluation.generation.evaluate", "_partial_": True,
                        "evaluation_name": "diversity", "indices": None}
    prepared = prepare_run({"config": OmegaConf.to_container(config, resolve=False)},
                           destination, {"GPU": 1})
    run(prepared)
    result = json.loads((destination / "results.json").read_text())
    if result["max_nfe"] > case["nfe"]:
        raise ValueError("sampler exceeded the NFE budget")


def summary(out_dir):
    rows = []
    for f in sorted(Path(out_dir).glob("*/*/f*/results.json")):
        case, sampler, frac = f.parts[-4], f.parts[-3], float(f.parts[-2][1:])
        r = json.loads(f.read_text())
        hints = [p["retained_hints"] for p in r["prompts"]]
        rows.append(dict(case=case, sampler=sampler, fraction=frac,
                         mean_hints=sum(hints) / len(hints), valid_rate=r["valid_rate"],
                         valid_unique_per_puzzle=r["mean_valid_unique_per_prompt"],
                         unique_per_puzzle=r["mean_unique_per_prompt"],
                         pooled_valid_unique=r["pooled_valid_unique_boards"],
                         generations=r["num_generations"]))
    return rows


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "summary":
        rows = summary(sys.argv[2])
        Path(sys.argv[2], "summary.json").write_text(json.dumps(rows, indent=1))
        print(len(rows), "jobs")
        sys.exit()
    spec = json.loads(Path(sys.argv[2]).read_text())
    if mode == "jobs":
        print("\n".join(job_ids(spec)))
    elif mode == "run":
        run_job(spec, sys.argv[3], sys.argv[4])
