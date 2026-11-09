"""Run every case of the given manifests on this machine's GPUs with one local Ray cluster.

Used by the self-healing spot runner (``run.sh``); safe to rerun after an interruption:
nested tuning reuses finished trials and continues half-trained candidates from last.ckpt.
Exit status 0 only when every case has its final result.

    python run_all.py --runs /data/sddc/runs/RUN PLAN_DIR [PLAN_DIR ...]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def cases(plans):
    for plan in plans:
        for spec in json.loads((Path(plan) / "manifest.json").read_text()):
            yield Path(plan).name, spec


def done(runs, plan, spec):
    return (Path(runs) / plan / spec["name"] / "final" / "result.json").is_file()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--case-concurrency", type=int, default=None, help="Default: number of GPUs")
    parser.add_argument("--concurrency", type=int, default=2, help="Inference candidates per case")
    parser.add_argument("--cases", type=Path, help="Run only these PLAN/CASE names (one per line)")
    args = parser.parse_args()
    import ray
    import torch

    from core.sweeps import nested_tuning

    gpus = torch.cuda.device_count()
    storage = os.environ["SDDC_STORAGE_ROOT"]
    env = {k: os.environ[k] for k in ("SDDC_STORAGE_ROOT", "TINYSTORIES_DATA_DIR",
                                      "SDDC_SOURCE_ID", "SDDC_SOURCE_REV", "SDDC_SOURCE_DIRTY")
           if k in os.environ}
    ray.init(address="local", num_gpus=gpus, include_dashboard=False,
             _temp_dir=f"{storage}/ray",
             runtime_env={"env_vars": {"PYTHONPATH": str(ROOT), **env}})
    owned = None if args.cases is None else set(args.cases.read_text().split())
    selected = [(p, s) for p, s in cases(args.plans) if owned is None or f"{p}/{s['name']}" in owned]
    pending = [(plan, spec) for plan, spec in selected if not done(args.runs, plan, spec)]
    print(f"{len(pending)} cases to run on {gpus} GPUs", flush=True)

    def one(item):
        plan, spec = item
        try:
            nested_tuning.run_case(spec, args.runs / plan, args.concurrency, retry_skipped=True)
            print(f"done {plan}/{spec['name']}", flush=True)
        except Exception as error:  # noqa: BLE001 - keep the other cases running
            print(f"FAILED {plan}/{spec['name']}: {error!r}", flush=True)

    with ThreadPoolExecutor(args.case_concurrency or max(gpus, 1)) as pool:
        list(pool.map(one, pending))
    remaining = [f"{p}/{s['name']}" for p, s in selected if not done(args.runs, p, s)]
    print(f"{len(remaining)} cases without a final result", flush=True)
    raise SystemExit(1 if remaining else 0)


if __name__ == "__main__":
    main()
