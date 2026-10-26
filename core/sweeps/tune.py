"""Run the existing training/evaluation lifecycle as a Ray Tune function."""

import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from omegaconf import OmegaConf

from core.evaluation.reporting import scalar_metrics
from core.sweeps.configuration import ROOT, compose


def _configuration(spec):
    """Accept either a Hydra preset specification or an editable config snapshot."""
    unknown = set(spec) - {"experiment", "overrides", "config"}
    if unknown:
        raise ValueError(f"Unknown trial options: {sorted(unknown)}")
    if "config" in spec:
        if "experiment" in spec or "overrides" in spec:
            raise ValueError("Use either config or experiment/overrides, not both")
        return OmegaConf.create(spec["config"])
    return compose(spec.get("experiment"), spec.get("overrides", {}))


def _prepare(spec, output, resources):
    config = _configuration(spec)
    gpus = resources.get("GPU", 0)
    if (
        not isinstance(gpus, (int, float))
        or not 0 <= gpus < float("inf")
        or int(gpus) != gpus
    ):
        raise ValueError("run_trial requires a nonnegative whole GPU allocation")
    gpus = int(gpus)
    if config.trainer.get("num_nodes", 1) != 1:
        raise ValueError("run_trial supports one node per trial")
    strategy = config.get("strategy")
    if gpus > 1:
        if config.mode != "train":
            raise ValueError("Multi-GPU trials currently support training only")
        if config.trainer.strategy not in (
            None,
            "auto",
            "ddp",
            "ddp_find_unused_parameters_true",
            "ddp_find_unused_parameters_false",
        ):
            raise ValueError("Multi-GPU trials require a DDP strategy")
        if strategy is not None and (
            strategy.get("_target_") != "lightning.pytorch.strategies.DDPStrategy"
            or strategy.get("start_method", "popen") != "popen"
        ):
            raise ValueError(
                "Multi-GPU trials require a DDP strategy with start_method=popen"
            )
        if config.trainer.strategy in (None, "auto"):
            config.trainer.strategy = "ddp"
    elif config.trainer.strategy not in ("auto", None) or strategy is not None:
        raise ValueError(
            "Zero/one-GPU trials require strategy=auto and no strategy config"
        )
    # The Tune reservation is authoritative; never use unreserved devices.
    config.trainer.devices = max(1, gpus)
    config.output_dir = str(Path(output).resolve())
    config.trainer.accelerator = "gpu" if gpus else "cpu"
    config.evaluation.device = "cuda:0" if gpus else "cpu"
    # Freeze interpolations only after assigning the per-trial output path.
    return OmegaConf.create(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )


def _run_distributed(config):
    """Run ranks outside the Tune actor; only that actor reports to Tune."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "distributed_config.yaml"
    OmegaConf.save(config, path)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={config.trainer.devices}",
        "--module",
        "core.sweeps.distributed_worker",
        str(path),
    ]
    # Children inherit Ray's CUDA_VISIBLE_DEVICES, including noncontiguous IDs.
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def _metrics(output):
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
                    if value:
                        if value in {"True", "False"}:
                            result[key] = value == "True"
                        else:
                            result[key] = float(value)
    return result


def run_trial(spec):
    """Tune trainable; use with_resources, Tuner and grid_search directly.

    ``spec`` is {experiment, overrides} or {config: plain_config_dict}. Wrap
    either form in {trial: ...} to grid-search a list of complete specifications.
    Reports final metrics only; retries resume the latest local training checkpoint.
    Set GPU count with tune.with_resources(run_trial, {"gpu": N, "cpu": ...}).
    N > 1 launches single-node DDP via torchrun using only the reserved GPUs.
    Batch size and gradient accumulation remain explicit training settings.
    """
    from ray import tune

    context = tune.get_context()
    if "trial" in spec:
        if set(spec) != {"trial"}:
            raise ValueError("Use trial alone; put experiment/overrides inside it")
        spec = spec["trial"]
    output = Path(context.get_trial_dir()).resolve() / "run"
    config = _prepare(spec, output, context.get_trial_resources().required_resources)
    # Read shared control state only at startup, so cancelling pending work
    # never interrupts a running trainer. Fresh Tune actors import this module.
    storage = context.get_storage()
    control = Path(storage.experiment_fs_path) / "queue-control.json"
    if control.is_file():
        cancelled = json.loads(control.read_text()).get("cancelled_wandb_ids", [])
        run_id = OmegaConf.select(config, "wandb.id")
        if run_id and run_id in cancelled:
            tune.report({"cancelled": True, "cancelled_run_id": run_id})
            return
    latest = output / "checkpoints" / "last.ckpt"
    if config.mode == "train" and latest.is_file():
        requested = config.checkpoint.resume
        if requested and Path(requested).is_file() and Path(requested) != latest:
            import torch

            def saved_step(path):
                return torch.load(
                    path, map_location="cpu", weights_only=False, mmap=True
                ).get("global_step", 0)

            # A live migration may provide a newer best/periodic checkpoint.
            if saved_step(latest) >= saved_step(requested):
                config.checkpoint.resume = str(latest)
        else:
            config.checkpoint.resume = str(latest)
    if config.trainer.accelerator == "cpu":
        # Ray can retain the driver's visibility on a CPU-only local cluster.
        # Auxiliary evaluators must not pick up an unreserved GPU either.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Ray already isolated this worker's visible GPUs. Preserve repository-relative
    # dataset/checkpoint paths while all writes go to the absolute trial directory.
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        if config.trainer.devices > 1:
            _run_distributed(config)
        else:
            from core.training.runner import run

            run(config)
    finally:
        os.chdir(previous_cwd)
    tune.report(
        {
            **_metrics(output),
            "output_dir": str(output),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            # Tune adds checkpoint_dir_name only after persistence succeeds.
            "artifact_root": context.get_storage().trial_fs_path,
            "artifact_filesystem": context.get_storage().storage_filesystem.type_name,
        },
        checkpoint=tune.Checkpoint.from_directory(output),
    )
