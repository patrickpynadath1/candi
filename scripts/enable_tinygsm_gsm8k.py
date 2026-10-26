"""One-time live queue migration: checkpoint, backfill GSM8K, restore with callbacks."""

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import psutil
from ray.job_submission import JobSubmissionClient, JobStatus

ROOT = Path("/scratch/gautschi/ppynadat/cont_diff/candiv2.1/tinygsm-ray-sc-20261013")
REPO = Path("/home/ppynadat/candi")
JOB = "tinygsm-queue-17020581-0"


def wait_job(client, job):
    while True:
        status = client.get_job_status(job)
        if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
            if status != JobStatus.SUCCEEDED:
                raise RuntimeError(f"{job}: {status}; inspect Ray logs")
            return
        time.sleep(10)


def main():
    client = JobSubmissionClient("http://127.0.0.1:8265")
    candidates = list(
        Path(os.environ["RAY_TMPDIR"]).glob(
            "session_*/artifacts/*/training/driver_artifacts/*/run/config.yaml"
        )
    )
    candidates = sorted({path.resolve() for path in candidates})
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one active run, got {len(candidates)}")
    source = candidates[0].parent
    import torch

    saved_checkpoints = []
    for path in (source / "checkpoints").glob("*.ckpt"):
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        saved_checkpoints.append((int(saved["global_step"]), path))
        del saved
    step, checkpoint = max(saved_checkpoints, key=lambda item: item[0])
    print(f"Latest saved checkpoint: {checkpoint.name}, step {step}", flush=True)
    processes = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        cmd = process.info["cmdline"] or []
        if (
            ("torch.distributed.run" in cmd or "core.sweeps.distributed_worker" in cmd)
            and any(str(source) in arg for arg in cmd)
            and os.getpgid(process.pid) == process.pid
        ):
            processes.append((process.pid, process.create_time()))
    print(f"Stopping original driver; resuming saved step {step}", flush=True)
    client.stop_job(JOB)
    for _ in range(60):
        if client.get_job_status(JOB) == JobStatus.STOPPED:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Original driver did not stop")
    # torchrun and its ranks use separate groups; clean up survivors of actor stop.
    for pid, created in processes:
        if psutil.pid_exists(pid) and psutil.Process(pid).create_time() == created:
            if os.getpgid(pid) != pid:
                raise RuntimeError("Unexpected torchrun process group")
            os.killpg(pid, signal.SIGTERM)
    time.sleep(10)
    durable = ROOT / "training" / source.parent.name / "run"
    durable.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", str(source / "config.yaml"), str(durable) + "/"], check=True
    )
    subprocess.run(
        ["rsync", "-a", str(source / "checkpoints"), str(durable) + "/"], check=True
    )
    (ROOT / "gsm8k-migration.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "durable": str(durable),
                "resume_checkpoint": str(durable / "checkpoints" / checkpoint.name),
            },
            indent=2,
        )
    )
    output = ROOT / "gsm8k-masked-checkpoints"
    evaluation = client.submit_job(
        submission_id="tinygsm-masked-gsm8k-backfill-17020581",
        entrypoint=shlex.join(
            [
                sys.executable,
                str(REPO / "scripts/evaluate_tinygsm_checkpoints.py"),
                "--run",
                str(source),
                "--output",
                str(output),
            ]
        ),
        entrypoint_num_cpus=0,
    )
    print(f"Backfill submitted: {evaluation}", flush=True)
    # Restore training even if an evaluation fails; surface the backfill failure separately.
    try:
        wait_job(client, evaluation)
    finally:
        training = client.submit_job(
            submission_id="tinygsm-queue-gsm8k-17020581",
            entrypoint=shlex.join(
                [
                    sys.executable,
                    str(REPO / "scripts/tinygsm_ray_queue.py"),
                    "--root",
                    str(ROOT),
                    "--resume-masked-from",
                    str(durable / "checkpoints" / checkpoint.name),
                ]
            ),
            entrypoint_num_cpus=0,
        )
        print(f"Training restored with periodic GSM8K: {training}", flush=True)


if __name__ == "__main__":
    main()
