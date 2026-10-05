"""Run the existing training/evaluation lifecycle as a Ray Tune function."""

import csv
import json
import os
from pathlib import Path

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
    if gpus not in (0, 1):
        raise ValueError("run_trial supports zero or one whole GPU per trial")
    if (
        config.trainer.devices != 1
        or config.trainer.get("num_nodes", 1) != 1
        or config.trainer.strategy not in ("auto", None)
        or config.get("strategy") is not None
    ):
        raise ValueError(
            "run_trial requires trainer.devices=1, num_nodes=1 and strategy=auto; "
            "distributed training within a Tune trial is not supported"
        )
    config.output_dir = str(Path(output).resolve())
    config.trainer.accelerator = "gpu" if gpus else "cpu"
    config.evaluation.device = "cuda:0" if gpus else "cpu"
    # Freeze interpolations only after assigning the per-trial output path.
    return OmegaConf.create(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )


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
                        result[key] = float(value)
    return result


def run_trial(spec):
    """Tune trainable; use with_resources, Tuner and grid_search directly.

    ``spec`` is {experiment, overrides} or {config: plain_config_dict}. Wrap
    either form in {trial: ...} to grid-search a list of complete specifications.
    Reports final metrics only; retries restart unless checkpoint.resume is set.
    """
    from ray import tune

    context = tune.get_context()
    if "trial" in spec:
        if set(spec) != {"trial"}:
            raise ValueError("Use trial alone; put experiment/overrides inside it")
        spec = spec["trial"]
    output = Path(context.get_trial_dir()).resolve() / "run"
    config = _prepare(spec, output, context.get_trial_resources().required_resources)
    if config.trainer.accelerator == "cpu":
        # Ray can retain the driver's visibility on a CPU-only local cluster.
        # Auxiliary evaluators must not pick up an unreserved GPU either.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Ray already isolated this worker's visible GPUs. Preserve repository-relative
    # dataset/checkpoint paths while all writes go to the absolute trial directory.
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
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
