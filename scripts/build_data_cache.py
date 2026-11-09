"""Build every data cache the worked-example sweeps read, on a CPU machine; no training.

Prepares each sweep's full grid into a plan under ``$SDDC_STORAGE_ROOT/data-build/plans``,
then calls ``task.prepare_data()`` once for each distinct task config in those plans, in
parallel. Caches land where training looks for them (their names hash their inputs, not the
machine), so a GPU machine that syncs ``data_cache/`` reuses them as-is. Needs the text8
splits and the TinyStories cohort first (``scripts/prepare_data.py``), and
``TINYSTORIES_DATA_DIR`` set as it will be for training.

    python scripts/build_data_cache.py [--sweep sc_vs_depth ...] [--workers 8]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SWEEPS = ["sc_vs_depth", "continuous_schedules", "block_encoder_decoder", "tokenizer_ladder"]


def plan(sweep, output):
    if not (output / "manifest.json").exists():
        subprocess.run([sys.executable, "-m", "core.sweeps.prepare", "-m", f"+sweep={sweep}",
                        f"preparation.output={output}"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "core.sweeps.prepare", "collect", str(output)],
                       cwd=ROOT, check=True)
    return json.loads((output / "manifest.json").read_text())


def build(config_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    start = time.time()
    task = instantiate(OmegaConf.load(config_path).task)
    task.prepare_data()
    return round((time.time() - start) / 60, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="append", choices=SWEEPS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() // 4 or 1)
    args = parser.parse_args()
    from omegaconf import OmegaConf

    storage = Path(os.environ.get("SDDC_STORAGE_ROOT", Path.home() / ".cache" / "sddc"))
    tasks = {}  # resolved task config -> one case config that uses it
    for sweep in args.sweep or SWEEPS:
        for case in plan(sweep, storage / "data-build" / "plans" / sweep):
            task = OmegaConf.load(case["config"]).task
            key = json.dumps(OmegaConf.to_container(task, resolve=True), sort_keys=True)
            tasks.setdefault(hashlib.sha256(key.encode()).hexdigest()[:12], (case["name"], case["config"]))
    print(f"{len(tasks)} distinct task configs", flush=True)
    failed = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(build, config): name for name, config in tasks.values()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                print(f"ok   {name} ({future.result()} min)", flush=True)
            except Exception as error:  # noqa: BLE001 - report every task, then fail
                failed.append(name)
                print(f"FAIL {name}: {error!r}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
