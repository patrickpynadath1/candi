"""Local subprocess scheduling and durable sweep state (independent of composition)."""

import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf

from core.evaluation.reporting import scalar_metrics

ROOT = Path(__file__).resolve().parents[2]


class LocalExecutor:
    """One-device-per-trial queue; configurations must already be resolved."""

    def __init__(self, output_dir, resources):
        self.output_dir = Path(output_dir).resolve()
        self.resources = resources

    def run(self, plans, *, resume=True):
        slots = self.resources.slots()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / ".lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "Another coordinator is using this sweep directory"
                ) from error
            return self._execute(plans, slots, resume)

    def _execute(self, plans, slots, resume):
        manifest_path = self.output_dir / "manifest.json"
        previous = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        )
        identity = [(p["name"], p["fingerprint"]) for p in plans]
        if previous:
            if not resume:
                raise ValueError(
                    "Sweep directory already has a manifest; choose a new output_dir or resume=True"
                )
            if identity != [(p["name"], p["fingerprint"]) for p in previous["trials"]]:
                raise ValueError(
                    "Sweep configs differ from existing manifest; choose a new output_dir"
                )
        if not previous:
            occupied = [
                p["output_dir"] for p in plans if Path(p["output_dir"]).exists()
            ]
            if occupied:
                raise ValueError(
                    f"Trial output already exists without a matching manifest: {occupied[0]}"
                )
        rows = (
            previous["trials"]
            if previous
            else [dict(p, status="pending", returncode=None, metrics={}) for p in plans]
        )
        manifest = dict(version=1, trials=rows)

        def save():
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(manifest_path)

        pending = [row for row in rows if row["status"] != "succeeded"]
        active = []
        save()
        try:
            while pending or active:
                while pending and slots:
                    row, device = pending.pop(0), slots.pop(0)
                    output = Path(row["output_dir"])
                    output.mkdir(parents=True, exist_ok=True)
                    config_path = output / "trial.yaml"
                    OmegaConf.save(OmegaConf.create(row["config"]), config_path)
                    env = os.environ.copy()
                    env.update(
                        CUDA_VISIBLE_DEVICES="" if device is None else device,
                        OMP_NUM_THREADS=str(self.resources.cpus),
                        MKL_NUM_THREADS=str(self.resources.cpus),
                    )
                    env["PYTHONPATH"] = (
                        str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                    )
                    log = (output / "run.log").open("a")
                    try:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "core.sweeps.worker",
                                str(config_path),
                            ],
                            cwd=ROOT,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except Exception:
                        log.close()
                        raise
                    row.update(
                        status="running", returncode=None, metrics={}, gpu=device
                    )
                    active.append((process, log, row, device))
                    save()
                for process, log, row, device in active[:]:
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    row.update(
                        status="succeeded" if code == 0 else "failed", returncode=code
                    )
                    if code == 0:
                        row["metrics"] = _metrics(Path(row["output_dir"]))
                    active.remove((process, log, row, device))
                    slots.append(device)
                    save()
                if active:
                    time.sleep(0.1)
        finally:
            for process, _, _, _ in active:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process, log, row, device in active:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                log.close()
                row.update(status="interrupted", returncode=process.returncode)
            save()
        return rows


def _metrics(output: Path) -> dict:
    if (output / "results.json").exists():
        return dict(scalar_metrics(json.loads((output / "results.json").read_text())))
    files = sorted(
        (output / "metrics").glob("version_*/metrics.csv"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    result = {}
    if files:
        with files[-1].open() as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if value == "":
                        continue
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
    return result
