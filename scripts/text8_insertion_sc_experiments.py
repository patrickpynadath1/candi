"""Run the four single-view, self-conditioned Text8 insertion experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("gaussian", "masked", "simplex", "spherical")


def trial(kind, gpu, cache, *, experiment_prefix="text8_insertion"):
    name = f"{experiment_prefix}_{kind}_sc_22m"
    output = ROOT / "core/outputs" / name
    if (output / "config.yaml").exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")

    def status(stage, **details):
        (output / "status.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "gpu": gpu,
                    "experiment": name,
                    "updated_at_unix": time.time(),
                    **details,
                },
                indent=2,
            )
        )
        print(name, stage, details, flush=True)

    def run(destination, *overrides):
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "main.py",
            "+experiment=" + name,
            "output_dir=" + str(destination),
            "data.cache_dir=" + cache,
            *overrides,
        ]
        with (destination / "train.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            (destination / "launch.json").write_text(
                json.dumps({"pid": process.pid, "argv": command, "gpu": gpu}, indent=2)
            )
            code = process.wait()
            if code:
                raise RuntimeError(
                    f"process {process.pid} exited {code}; see {destination / 'train.log'}"
                )

    try:
        status("training")
        run(output)
        status("evaluating")
        run(
            output / "test_temp0p7_seed23",
            "mode=evaluate",
            "seed=23",
            "task.evaluation_split=test",
            "evaluation.checkpoint=" + str(output / "checkpoints/last.ckpt"),
        )
        status("complete", results=str(output / "test_temp0p7_seed23/results.json"))
        return True
    except Exception as exc:
        status("failed", error=str(exc))
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    args = parser.parse_args()
    if len(set(args.gpus)) != 4:
        parser.error("assign a distinct GPU to each experiment")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(trial, kind, gpu, args.cache_dir)
            for kind, gpu in zip(KINDS, args.gpus)
        ]
        outcomes = [job.result() for job in jobs]
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
