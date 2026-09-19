#!/usr/bin/env python3
"""Run the S-FLM-schedule hybrid SLERP Sudoku Hard comparison."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_reasoning_method_smoke import snapshot


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/sudoku_hard_hybrid_slerp_sflm_20k_20260919"
DATA_CACHE = ROOT / "outputs/sudoku_core_ablation_20k_seed1_20260915/data"
RUN_NAME = "sudoku_hard_hybrid_slerp_sflm_schedule_20k_seed1"
RUN_ID = hashlib.sha256(RUN_NAME.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (CAMPAIGN / "supervisor.log").open("a") as output:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--gpu", args.gpu],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(process.pid)
        return

    source = snapshot(CAMPAIGN)
    run_dir = CAMPAIGN / "run"
    run_dir.mkdir(exist_ok=True)
    url = f"https://wandb.ai/rz-text-dlp/reasoning-sudoku-core-ablations/runs/{RUN_ID}"
    overrides = [
        "+experiment=sudoku_candi_slerp_sflm_schedule",
        "+callbacks=reasoning_eval",
        "seed=1",
        f"data.cache_dir={DATA_CACHE}",
        "model.length=180",
        "model.hidden_size=512",
        "loader.global_batch_size=256",
        "loader.eval_global_batch_size=128",
        "loader.num_workers=4",
        "reasoning.batch_size=64",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.accumulate_grad_batches=1",
        "trainer.max_steps=20000",
        "trainer.num_sanity_val_steps=0",
        "callbacks.reasoning_eval.every_n_train_steps=1000",
        "callbacks.reasoning_eval.subset_size=128",
        "callbacks.reasoning_eval.num_steps=180",
        "callbacks.checkpoint_every_n_steps.every_n_train_steps=5000",
        "callbacks.checkpoint_every_n_steps.save_top_k=0",
        "callbacks.checkpoint_every_n_steps.save_last=true",
        "callbacks.checkpoint_monitor.save_top_k=0",
        "callbacks.checkpoint_monitor.monitor=null",
        "checkpointing.resume_from_ckpt=true",
        f"checkpointing.save_dir={run_dir}",
        f"hydra.run.dir={run_dir}",
        "wandb.project=reasoning-sudoku-core-ablations",
        "+wandb.entity=rz-text-dlp",
        f"wandb.id={RUN_ID}",
        f"wandb.name={RUN_NAME}",
        "wandb.group=sudoku_hard_sflm_schedule_20k",
        "wandb.job_type=train",
        "wandb.tags=[sudoku_hard,hybrid_slerp,sflm_schedule,seed1]",
    ]
    command = [sys.executable, str(source / "main.py"), *overrides]
    manifest = {
        "name": RUN_NAME,
        "wandb_url": url,
        "source": str(source),
        "gpu": args.gpu,
        "command": command,
        "schedule": "S-FLM truncated SLERP, alpha_min=0.001, alpha_max calculated from dimension=512 and delta=0.1; single adaptive clock also controls discrete reveal probability",
    }
    (CAMPAIGN / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="online",
        WANDB_RESUME="allow",
        PYTHONUNBUFFERED="1",
    )
    status = CAMPAIGN / "status.json"
    status.write_text(json.dumps({"state": "running", "pid": os.getpid(), "url": url}) + "\n")
    with (run_dir / "train.log").open("a") as output:
        code = subprocess.call(command, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT)
    result = {"state": "complete" if code == 0 else "failed", "returncode": code,
              "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": url}
    status.write_text(json.dumps(result, indent=2) + "\n")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
