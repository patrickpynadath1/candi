"""Short GPU check of the worked-example sweeps; run on the GPU server as a Ray Job.

For each sweep it (1) prepares the full published grid and validates it with
``nested_tuning --check`` (no training), then (2) prepares a subset that covers every recipe
arm and every task once, and runs each of those cases through nested tuning with real model
sizes, data and sampler policies but a few optimizer steps, a small NFE budget and one
trial per policy. Results go to ``$SDDC_STORAGE_ROOT/gpu-check/<tag>/``; a case that fails
is recorded and the next one still runs. Exit status is non-zero if anything failed.

    python scripts/validation/gpu_check.py --tag 20261109 [--only sc_vs_depth] [--steps 5]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# (sweep, subset name, overrides). Each sweep: all arms on one task + one arm on the other tasks.
# Sudoku and text8 subsets first; TinyStories ones (which need the prepared cohort) last.
SUBSETS = [
    ("sc_vs_depth", "arms", ["task=battery/sudoku_hard"]),
    ("continuous_schedules", "arms", ["task=battery/sudoku_hard"]),
    ("block_encoder_decoder", "arms", ["task=battery/text8"]),
    ("sc_vs_depth", "tasks", ["task=battery/text8,battery/tinystories_bpe",
                              "ablation_path=masked_elbo", "modifiers/feedback=none"]),
    ("continuous_schedules", "tasks", ["task=battery/text8,battery/tinystories_bpe",
                                       "ablation_path=gaussian", "modifiers/feedback=none"]),
    ("block_encoder_decoder", "tasks", ["task=battery/sudoku_hard,battery/tinystories_bpe",
                                        "denoising_task/block=encoder8_decoder8"]),
    ("tokenizer_ladder", "arms", ["bpe.vocabulary_size=10000"]),
    # The longest sequences (small vocabularies) with SC are the most memory-hungry cells.
    ("tokenizer_ladder", "tasks", ["bpe.vocabulary_size=260,500,1000,5000",
                                   "ladder_recipe=gaussian_sc"]),
]


def short(steps, nfe=True):
    # Block diffusion needs two network calls per block, so keep its sweep budget.
    budget = ["preparation.evaluation.nfe=8"] if nfe else []
    return [*budget,
        f"trainer.max_steps={steps}", f"trainer.val_check_interval={steps}",
        "trainer.limit_val_batches=2", "trainer.log_every_n_steps=1",
        f"checkpoint.every_n_steps={steps}", "callback.every_n_steps=0",
        "training.warmup_steps=1",
        "preparation.evaluation.seeds=[41]", "preparation.evaluation.test_seeds=[1041]",
        "+preparation.evaluation.tuning_indices=[0,1]",
        "+preparation.evaluation.test_indices=[2,3]",
    ]


def prepare(sweep, output, overrides):
    command = [sys.executable, "-m", "core.sweeps.prepare", "-m", f"+sweep={sweep}",
               f"preparation.output={output}", *overrides]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"prepare failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    subprocess.run([sys.executable, "-m", "core.sweeps.prepare", "collect", str(output)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    return output / "manifest.json"


def check(manifest):
    result = subprocess.run([sys.executable, "-m", "core.sweeps.nested_tuning", str(manifest),
                             "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="Names the output folder; use a new one per check")
    parser.add_argument("--only", action="append", help="Restrict to these sweeps")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--skip-full-check", action="store_true")
    parser.add_argument("--subset", action="append", metavar="SWEEP|NAME|OVERRIDES",
                        help="Replace the built-in subsets, e.g. 'tokenizer_ladder|ar|ladder_recipe=ar'")
    parser.add_argument("--keep-nfe", action="store_true",
                        help="Keep each sweep's NFE budget (needed by block diffusion and AR)")
    args = parser.parse_args()
    storage = Path(os.environ.get("SDDC_STORAGE_ROOT", Path.home() / ".cache" / "sddc"))
    out = storage / "gpu-check" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    def save():
        summary_path.write_text(json.dumps(summary, indent=2))

    subsets = [(s, n, o.split()) for s, n, o in (x.split("|") for x in args.subset)] if args.subset else SUBSETS
    sweeps = [s for s in dict.fromkeys(s for s, _, _ in subsets) if not args.only or s in args.only]

    # 1. The full published grids, validated without training.
    if not args.skip_full_check:
        for sweep in sweeps:
            key = f"full/{sweep}"
            if summary.get(key, {}).get("ok"):
                continue
            try:
                manifest = prepare(sweep, out / "plans" / f"{sweep}-full", [])
                ok, log = check(manifest)
                summary[key] = {"ok": ok, "cases": len(json.loads(manifest.read_text())), "log": log}
            except Exception as error:  # noqa: BLE001 - record and continue
                summary[key] = {"ok": False, "log": str(error)[-4000:]}
            print(f"[full] {sweep}: {'ok' if summary[key]['ok'] else 'FAILED'}", flush=True)
            save()

    # 2. Short GPU runs of the subsets.
    import ray

    from core.sweeps import nested_tuning

    ray.init(address="auto")
    for sweep, name, overrides in subsets:
        if sweep not in sweeps:
            continue
        try:
            manifest = prepare(sweep, out / "plans" / f"{sweep}-{name}", [*overrides, *short(args.steps, not args.keep_nfe and sweep != "block_encoder_decoder")])
        except Exception as error:  # noqa: BLE001
            summary[f"prepare/{sweep}-{name}"] = {"ok": False, "log": str(error)[-4000:]}
            print(f"[prepare] {sweep}-{name}: FAILED", flush=True)
            save()
            continue
        for case in json.loads(manifest.read_text()):
            key = f"run/{sweep}/{case['name']}"
            if summary.get(key, {}).get("ok"):
                continue
            for policy in case["inference"]:
                policy["trials"] = 1  # One sampling trial per policy is enough to exercise it.
            start = time.time()
            try:
                nested_tuning.run_case(case, out / "runs" / sweep, concurrency=1)
                final = out / "runs" / sweep / case["name"] / "final" / "result.json"
                ok = final.is_file()
                summary[key] = {"ok": ok, "log": "" if ok else "no final/result.json"}
            except Exception:  # noqa: BLE001
                summary[key] = {"ok": False, "log": traceback.format_exc()[-4000:]}
            summary[key]["minutes"] = round((time.time() - start) / 60, 2)
            print(f"[run] {sweep}/{case['name']}: {'ok' if summary[key]['ok'] else 'FAILED'} "
                  f"({summary[key]['minutes']} min)", flush=True)
            save()

    failed = [k for k, v in summary.items() if not v["ok"]]
    print(f"\n{len(summary) - len(failed)} ok, {len(failed)} failed")
    for key in failed:
        print(f"  FAILED {key}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
