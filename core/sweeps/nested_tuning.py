"""Validation-only Optuna searches with Ray workers and one final evaluation.

A fixed checkpoint is the inference-only case. An optional training search adds
an outer loop; each trained checkpoint receives the same inference search budget.
The driver owns both studies and holds no GPUs while waiting for child work.
"""

import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import logging
import os
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from core.evaluation.reporting import metric_value
from core.sweeps.configuration import prepare_run


@contextmanager
def lock(path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


# Recorded in protocol.json but not compared on resume: where a case runs and
# where it logs do not change what is selected or scored.
EXECUTION_FIELDS = ("resources", "wandb")


def selection_protocol(protocol):
    """The part of a stored protocol (v1 or later) that a resume must match."""
    spec = {k: v for k, v in protocol["spec"].items() if k not in EXECUTION_FIELDS}
    config = {k: v for k, v in protocol["config"].items() if k != "wandb"}
    return {"spec": spec, "config": config}


def source_identity():
    """Snapshot that produced a result, as set by scripts/cluster/sddc_remote.py."""
    return {
        key: os.environ.get(f"SDDC_SOURCE_{key.upper()}")
        for key in ("id", "rev", "dirty")
    }


def suggest(trial, space):
    values = {}
    for key, distribution in space.items():
        if isinstance(distribution, list):
            values[key] = trial.suggest_categorical(key, distribution)
        elif distribution["type"] == "float":
            values[key] = trial.suggest_float(
                key,
                distribution["low"],
                distribution["high"],
                log=distribution.get("log", False),
            )
        elif distribution["type"] == "int":
            values[key] = trial.suggest_int(
                key,
                distribution["low"],
                distribution["high"],
                step=distribution.get("step", 1),
                log=distribution.get("log", False),
            )
        else:
            raise ValueError(f"Unknown distribution for {key}")
    return values


def validate_inference(spec):
    validate_search(spec)
    tune_ids, test_ids = spec["tuning_indices"], spec["test_indices"]
    same_dataset = spec.get("test_dataset", "validation") == spec.get(
        "dataset", "validation"
    )
    if not tune_ids or not test_ids or (same_dataset and set(tune_ids) & set(test_ids)):
        raise ValueError("Tuning and test indices must be nonempty and disjoint")
    for ids in (tune_ids, test_ids):
        if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Partition indices must be unique nonnegative integers")
    if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Specify unique generation seeds")
    if not spec["test_seeds"] or set(spec["seeds"]) & set(spec["test_seeds"]):
        raise ValueError(
            "Final generation seeds must be nonempty and separate from tuning"
        )
    if spec["nfe"] < 2 or spec["trials"] < 1:
        raise ValueError("Invalid NFE or trial budget")
    if spec["direction"] not in ("minimize", "maximize"):
        raise ValueError("Invalid optimization direction")
    if spec["budget_parameter"] not in (None, "steps", "max_steps"):
        raise ValueError("Unsupported sampler budget parameter")
    for key in spec["space"]:
        if key in {"sampler.steps", "sampler.max_steps"} or not (
            key.startswith("sampler.")
            or key in {"algorithm.predictor.temperature", "algorithm.predictor.commitment"}
        ):
            raise ValueError(f"Not a tunable inference parameter: {key}")


def validate(spec):
    training = spec["training"]
    if training.get("checkpoint") and (training["space"] or training["trials"] != 1):
        raise ValueError("An existing checkpoint is one fixed training candidate")
    if not spec["inference"] or len({s["name"] for s in spec["inference"]}) != len(
        spec["inference"]
    ):
        raise ValueError("Inference policies need unique names")
    evaluation = spec["evaluation"]
    if evaluation.get("dataset", "validation") != "validation":
        raise ValueError("Selection must use the validation dataset")
    if evaluation["score"]["kind"] != "metric":
        raise ValueError("Nested searches currently require a scalar task metric")
    for policy in spec["inference"]:
        validate_inference({**evaluation, **policy})
    training = spec["training"]
    validate_search(training)
    if training["trials"] < 1:
        raise ValueError("Training trial budget must be positive")
    for key in training["space"]:
        if not key.startswith(
            (
                "optimizer.",
                "model.",
                "algorithm.",
                "objective.",
                "noise_schedule.",
                "time_sampler.",
                "training.",
                "self_conditioning.",
                "lr_scheduler.",
            )
        ):
            raise ValueError(f"Not a training hyperparameter: {key}")
    for override in evaluation.get("overrides", {}):
        if not override.startswith(("task.", "loader.eval_batch_size")):
            raise ValueError(f"Not an evaluation override: {override}")
    resources = spec.get("resources", {"cpu": 4, "gpu": 1})
    if resources.get("gpu", 1) not in (0, 1) or resources.get("cpu", 4) < 1:
        raise ValueError("Workers require positive CPUs and zero or one whole GPU")


# Optuna samplers a study may use. The default reproduces the TPE search used by
# existing protocols; options are passed to the Optuna constructor unchanged.
SEARCH_ALGORITHMS = {
    "tpe": "TPESampler",
    "random": "RandomSampler",
    "qmc": "QMCSampler",
}
FAILURE_POLICIES = ("skip", "stop")


class StopSearch(RuntimeError):
    """A failure under on_failure=stop; an enclosing search never skips it."""


def make_sampler(settings, completed_trials=0):
    """Build a study's Optuna sampler from its ``search`` block.

    ``search: {algorithm: tpe|random|qmc, ...options}``; the seed comes from the
    study's ``seed`` (default 42). A resumed TPE or random study is reseeded by
    its trial count so it never replays proposals it already made. QMC indexes a
    fixed low-discrepancy sequence by trial number, so its seed stays fixed.
    """
    import optuna

    search = dict(settings.get("search") or {"algorithm": "tpe"})
    algorithm = search.pop("algorithm", "tpe")
    if algorithm not in SEARCH_ALGORITHMS:
        raise ValueError(
            f"Unknown search algorithm {algorithm!r}; use one of {sorted(SEARCH_ALGORITHMS)}"
        )
    if "seed" in search:
        raise ValueError("Set the study seed with `seed`, not inside `search`")
    seed = settings.get("seed", 42)
    if algorithm != "qmc":
        seed += completed_trials
    return getattr(optuna.samplers, SEARCH_ALGORITHMS[algorithm])(seed=seed, **search)


def validate_search(settings):
    if settings.get("on_failure", "skip") not in FAILURE_POLICIES:
        raise ValueError(f"on_failure must be one of {FAILURE_POLICIES}")
    algorithm = (settings.get("search") or {}).get("algorithm", "tpe")
    if algorithm not in SEARCH_ALGORITHMS:
        raise ValueError(
            f"Unknown search algorithm {algorithm!r}; use one of {sorted(SEARCH_ALGORITHMS)}"
        )
    try:
        import optuna  # noqa: F401
    except ImportError:  # Without the tuning extra, only the names are checked.
        return
    try:
        make_sampler(settings)
    except TypeError as error:
        raise ValueError(f"Invalid search options: {error}") from error


def search(directory, settings, direction, objective, concurrency, retry_skipped=False):
    """The same validation-only loop for training and inference candidates.

    With ``on_failure: skip`` (the default) a failed candidate is recorded in
    ``skipped.jsonl`` and ``selected.json`` and the search continues; it still
    counts against the trial budget and is not retried on resume. With
    ``on_failure: stop`` the first failure stops the search after running work
    finishes, and a rerun retries it. ``retry_skipped`` retries skipped
    candidates once more on this run. Returns None when every trial failed.
    """
    import optuna

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    skip = settings.get("on_failure", "skip") == "skip"
    log = logging.getLogger(__name__)
    study = optuna.create_study(
        storage=f"sqlite:///{directory / 'optuna.sqlite'}",
        study_name="search",
        direction=direction,
        load_if_exists=True,
        sampler=make_sampler(settings),
    )
    if study.trials:
        study.sampler = make_sampler(settings, len(study.trials))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    def still_skipped(trials):
        # A skipped candidate later retried successfully is no longer skipped.
        return [
            t
            for t in trials
            if t.user_attrs.get("skipped")
            and not any(c.params == t.params for c in completed)
        ]

    skipped = [] if retry_skipped else still_skipped(study.trials)
    if not study.trials:
        for point in settings.get("initial_points", []):
            study.enqueue_trial(point)

    def record_skip(number, parameters, error, output):
        entry = {
            "trial": number,
            "parameters": parameters,
            "error": f"{type(error).__name__}: {error}",
            "failure": str(output / "failure.json"),
        }
        with (directory / "skipped.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        log.warning("Skipped failed candidate %s in %s: %s", number, directory, entry["error"])
        return entry

    def candidate(trial, parameters):
        # Retry a completed stage without repeating training or generation.
        output = candidate_directory(parameters)
        result_path = output / "result.json"
        trial.set_user_attr("candidate", str(output))
        with lock(output / ".candidate.lock"):
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                try:
                    result = objective(parameters, output)
                    if not math.isfinite(result["score"]):
                        raise ValueError("Nonfinite validation score")
                except Exception:
                    # Preserve the actual worker exception even if the driver is
                    # waiting on other cases or this is a recovered trial.
                    save(
                        output / "failure.json",
                        {
                            "parameters": parameters,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise
                result = {**result, "source": source_identity()}
                save(result_path, result)
        trial.set_user_attr("result_path", str(result_path))
        return result["score"]

    def candidate_directory(parameters):
        key = hashlib.sha256(
            json.dumps(parameters, sort_keys=True).encode()
        ).hexdigest()[:20]
        return directory / "candidates" / key

    # Finish interrupted candidates before proposing new configurations. Trials
    # skipped under on_failure=skip stay skipped.
    for old in study.trials:
        if old.state == optuna.trial.TrialState.RUNNING:
            study.tell(old.number, state=optuna.trial.TrialState.FAIL)
        if old.state not in (
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.FAIL,
        ) or (old.user_attrs.get("skipped") and not retry_skipped):
            continue
        if set(old.params) != set(settings["space"]) or any(
            t.params == old.params for t in completed + skipped
        ):
            continue
        retry = optuna.trial.FixedTrial(old.params)
        parameters = suggest(retry, settings["space"])
        try:
            score = candidate(retry, parameters)
        except Exception as error:
            if not skip or isinstance(error, StopSearch):
                raise
            entry = record_skip(old.number, parameters, error, candidate_directory(parameters))
            marker = optuna.trial.create_trial(
                params=old.params,
                distributions=old.distributions,
                state=optuna.trial.TrialState.FAIL,
                user_attrs={"skipped": True, "error": entry["error"]},
            )
            study.add_trial(marker)
            skipped.append(marker)
            continue
        recovered = optuna.trial.create_trial(
            params=old.params,
            distributions=old.distributions,
            value=score,
            user_attrs=retry.user_attrs,
        )
        study.add_trial(recovered)
        completed.append(recovered)

    remaining = settings["trials"] - len(completed) - len(skipped)
    # SQLite cannot safely claim queued trials from multiple optimizer threads.
    # The driver owns trial allocation and completion; only evaluations run in parallel.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        failure = None
        while remaining > 0 or pending:
            while failure is None and remaining > 0 and len(pending) < concurrency:
                trial = study.ask()
                parameters = suggest(trial, settings["space"])
                pending[pool.submit(candidate, trial, parameters)] = (trial, parameters)
                remaining -= 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                trial, parameters = pending.pop(future)
                try:
                    score = future.result()
                except Exception as exc:
                    if skip and not isinstance(exc, StopSearch):
                        entry = record_skip(
                            trial.number, parameters, exc, candidate_directory(parameters)
                        )
                        trial.set_user_attr("skipped", True)
                        trial.set_user_attr("error", entry["error"])
                    else:
                        log.exception("Search candidate failed in %s", directory)
                        failure = failure or exc
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial, score)
            # Save results from already-running evaluations before surfacing an error.
            if failure is not None and not pending:
                if isinstance(failure, StopSearch):
                    raise failure
                raise StopSearch(f"{type(failure).__name__}: {failure}") from failure
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    skipped = [
        {"trial": t.number, "parameters": t.params, "error": t.user_attrs.get("error")}
        for t in still_skipped(study.trials)
    ]
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        if skip:
            log.warning("Every candidate failed in %s; see skipped.jsonl", directory)
            save(directory / "selected.json", {"selected": None, "skipped_trials": skipped})
            return None
        raise StopSearch(f"No completed candidates in {directory}")
    best = study.best_trial
    result = json.loads(Path(best.user_attrs["result_path"]).read_text())
    selected = {**result, "parameters": best.params, "skipped_trials": skipped}
    save(directory / "selected.json", selected)
    return selected


class ValidationTask:
    """Delegate task behavior while restricting all training-time validation reads."""

    def __init__(self, task, indices, dataset="validation"):
        from core.tasks.base import PrimaryMetric

        self.task, self.indices, self.dataset = task, list(indices), dataset
        self.primary_metric = PrimaryMetric(
            task.primary_metric.name, task.primary_metric.mode, dataset
        )

    def __getattr__(self, name):
        return getattr(self.task, name)

    def validation_dataset(self):
        from torch.utils.data import Subset

        return Subset(self.task.evaluation_dataset(self.dataset), self.indices)

    def evaluation_dataset(self, name):
        if name != self.dataset:
            raise ValueError(
                "Training callbacks may evaluate only the validation partition"
            )
        return self.validation_dataset()


def checkpoint_fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_checkpoint(checkpoint):
    """Fail if a recorded checkpoint file was replaced after it was scored."""
    expected = checkpoint.get("fingerprint")
    path = Path(checkpoint["checkpoint"])
    if expected is not None and (
        not path.is_file() or checkpoint_fingerprint(path) != expected
    ):
        raise ValueError(
            f"Checkpoint {checkpoint['checkpoint']} changed since it was recorded; "
            "use a new case name"
        )


def obtain_checkpoint(spec, config, parameters, validation, resources, output):
    """Checkpoint acquisition is the only distinction for a fixed outer candidate."""
    checkpoint = spec["training"].get("checkpoint")
    if checkpoint:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        return {"config": spec["config"], "checkpoint": checkpoint}
    return execute(
        "train",
        {
            "config": config,
            "parameters": parameters,
            "validation": validation,
            "resources": resources,
        },
        output,
    )


def _train_worker(payload, output):
    from core.training.runner import run

    config = OmegaConf.create(payload["config"])
    for key, value in payload["parameters"].items():
        if OmegaConf.select(config, key, default="__missing__") == "__missing__":
            raise ValueError(f"Unknown training parameter: {key}")
        OmegaConf.update(config, key, value, merge=False, force_add=False)
    for key, value in payload["validation"].get("overrides", {}).items():
        OmegaConf.update(config, key, value, force_add=True)
    config.mode = "train"
    config.checkpoint.resume = None
    config = prepare_run(
        {"config": OmegaConf.to_container(config, resolve=False)},
        output,
        {"GPU": payload["resources"].get("gpu", 1)},
    )
    if config.get("callback") is not None:
        config.callback.output_dir = str(Path(output) / "task_callback")
    # Keep periodic generation and validation loss confined to the same partition.
    evaluation_config = OmegaConf.to_container(config, resolve=True)
    config.task = {
        "_target_": "core.sweeps.nested_tuning.ValidationTask",
        "task": evaluation_config["task"],
        "indices": payload["validation"]["indices"],
        "dataset": payload["validation"].get("dataset", "validation"),
    }
    previous = Path(output) / "checkpoints/last.ckpt"
    completed = False
    if previous.exists():
        import torch

        config.checkpoint.resume = str(previous)
        # A worker can fail after saving the final checkpoint (e.g. logger
        # shutdown). Recover the handoff without constructing another trainer.
        saved = torch.load(previous, map_location="cpu", weights_only=False)
        max_steps = config.trainer.get("max_steps", -1)
        completed = max_steps > 0 and saved.get("global_step", -1) >= max_steps
        del saved
    if not completed:
        run(config)
    if not previous.is_file():
        raise FileNotFoundError(previous)
    evaluation_path = Path(output) / "evaluation-config.yaml"
    OmegaConf.save(OmegaConf.create(evaluation_config), evaluation_path)
    return {"config": str(evaluation_path), "checkpoint": str(previous)}


def _evaluate_worker(payload, output):
    from core.training.runner import run
    from hydra.utils import instantiate

    evaluation, policy = payload["evaluation"], payload["policy"]
    config = OmegaConf.load(payload["checkpoint"]["config"])
    for key, value in evaluation.get("overrides", {}).items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    config.sampler = OmegaConf.create(policy["sampler"])
    for key, value in payload["parameters"].items():
        # A tuned algorithm key that does not exist (e.g. a temperature on an
        # algorithm without a predictor) would make every trial identical.
        if key.startswith("algorithm.") and (
            OmegaConf.select(config, key, default="__missing__") == "__missing__"
        ):
            raise ValueError(f"Unknown inference parameter for this recipe: {key}")
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    # Resolve after trial overrides: correction count can itself be tuned.
    sampler = instantiate(config.sampler)
    from .budget import sampler_steps

    if policy["budget_parameter"] is not None:
        config.sampler[policy["budget_parameter"]] = sampler_steps(
            config, sampler, evaluation["nfe"]
        )
    config.mode = "evaluate"
    config.evaluation.checkpoint = payload["checkpoint"]["checkpoint"]
    config.evaluation.legacy = False
    config.evaluation.max_examples = None
    config.evaluator = {
        "_target_": "core.evaluation.generation.evaluate",
        "_partial_": True,
        "evaluation_name": evaluation.get("dataset", "validation"),
        "indices": evaluation["indices"],
    }
    scores = []
    for seed in evaluation["seeds"]:
        destination = Path(output) / f"seed-{seed}"
        config.seed = seed
        prepared = prepare_run(
            {"config": OmegaConf.to_container(config, resolve=False)},
            destination,
            {"GPU": payload["resources"].get("gpu", 1)},
        )
        run(prepared)
        result = json.loads((destination / "results.json").read_text())
        if result["max_nfe"] > evaluation["nfe"]:
            raise ValueError("Sampler exceeded NFE budget")
        scores.append(metric_value(result, evaluation["score"]["metric"]))
    return {"score": sum(scores) / len(scores), "seed_scores": scores}


def record_attempt(output):
    """Append this worker start to attempts.jsonl and return its number.

    Ray retries a task whose worker dies (e.g. an OOM kill), and a resumed
    driver re-runs unfinished stages; both appear here as extra attempts.
    """
    import datetime
    import socket

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entry = {
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": os.environ.get("SDDC_SOURCE_ID"),
    }
    try:
        import ray

        if ray.is_initialized():
            entry["ray_task_id"] = ray.get_runtime_context().get_task_id()
    except Exception:  # Attempt logging must never fail the worker.
        pass
    path = output / "attempts.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return sum(1 for _ in path.open())


def _run_worker(stage, payload, output):
    attempt = record_attempt(output)
    worker = _train_worker if stage == "train" else _evaluate_worker
    return {**worker(payload, output), "attempts": attempt}


def execute(stage, payload, output):
    import ray

    resources = payload["resources"]
    remote = ray.remote(
        num_cpus=resources.get("cpu", 4),
        num_gpus=resources.get("gpu", 1),
        max_calls=1,
        label_selector=resources.get("label_selector"),
    )(_run_worker)
    return ray.get(remote.remote(stage, payload, str(output)))


def run_case(spec, root, concurrency=4, retry_skipped=False):
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    path = Path(root).resolve() / spec["name"] / ".driver.lock"
    with lock(path, blocking=False):
        return _run_case(spec, root, concurrency, retry_skipped)


def case_protocol(spec):
    """What protocol.json records for a case: its spec and pinned config contents."""
    config = OmegaConf.to_container(OmegaConf.load(spec["config"]), resolve=False)
    if "wandb" in spec:
        config["wandb"] = spec["wandb"]
    return {"spec": spec, "config": config}


def protocol_conflict(protocol, path):
    """True when ``path`` holds a stored protocol a resume would refuse."""
    return path.exists() and selection_protocol(
        json.loads(path.read_text())
    ) != selection_protocol(protocol)


def _run_case(spec, root, concurrency, retry_skipped=False):
    validate(spec)
    spec = copy.deepcopy(spec)
    root = Path(root).resolve() / spec["name"]
    root.mkdir(parents=True, exist_ok=True)
    # Pin the configuration contents, not just its mutable filename.
    protocol = case_protocol(spec)
    config = protocol["config"]
    path = root / "protocol.json"
    if protocol_conflict(protocol, path):
        raise ValueError("Protocol changed; use a new case name")
    save(path, protocol)
    pinned_config = root / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), pinned_config)
    spec["config"] = str(pinned_config)
    resources = spec.get("resources", {"cpu": 4, "gpu": 1})
    direction = spec["evaluation"]["direction"]
    evaluation = spec["evaluation"]
    # Validation workers never receive final-test IDs or seeds.
    validation = {
        k: v
        for k, v in evaluation.items()
        if k not in ("test_indices", "test_seeds", "test_dataset", "tuning_indices")
    }
    validation["indices"] = evaluation["tuning_indices"]

    def inference(checkpoint, directory):
        choices = []
        for policy in spec["inference"]:

            def objective(parameters, output):
                result = execute(
                    "evaluate",
                    {
                        "checkpoint": checkpoint,
                        "policy": policy,
                        "parameters": parameters,
                        "evaluation": validation,
                        "resources": resources,
                    },
                    output,
                )
                return {
                    **result,
                    "checkpoint": checkpoint,
                    "policy": policy,
                    "inference_parameters": parameters,
                }

            choices.append(
                search(
                    directory / policy["name"],
                    policy,
                    direction,
                    objective,
                    concurrency,
                    retry_skipped,
                )
            )
        # A policy whose every trial failed (and was skipped) has no candidate.
        choices = [choice for choice in choices if choice is not None]
        if not choices:
            raise RuntimeError(
                f"Every inference policy failed for {checkpoint['checkpoint']}"
            )
        choose = max if direction == "maximize" else min
        return choose(choices, key=lambda r: r["score"])

    def train_candidate(parameters, output):
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output / "trained.json"
        recorded = checkpoint_path.exists()
        if recorded:
            checkpoint = json.loads(checkpoint_path.read_text())
            verify_checkpoint(checkpoint)
        else:
            checkpoint = obtain_checkpoint(
                spec, config, parameters, validation, resources, output / "training"
            )
            checkpoint["source"] = source_identity()
        # Pin the file actually scored; older records pin it at their first resume.
        pin = "fingerprint" not in checkpoint and Path(checkpoint["checkpoint"]).is_file()
        if pin:
            checkpoint["fingerprint"] = checkpoint_fingerprint(checkpoint["checkpoint"])
        if pin or not recorded:
            save(checkpoint_path, checkpoint)
        result = inference(checkpoint, output / "inference")
        return {**result, "training_parameters": parameters}

    winner = search(
        root / "training", spec["training"], direction, train_candidate, 1, retry_skipped
    )
    if winner is None:
        raise RuntimeError(
            f"Every training candidate failed for {spec['name']}; see skipped.jsonl files"
        )
    save(root / "selected.json", winner)
    # No final evaluation occurs until both levels and sampler selection finish.
    final_path = root / "final/result.json"
    if not final_path.exists():
        verify_checkpoint(winner["checkpoint"])
        final = {
            **validation,
            "dataset": evaluation.get(
                "test_dataset", validation.get("dataset", "validation")
            ),
            "indices": evaluation["test_indices"],
            "seeds": evaluation["test_seeds"],
        }
        result = execute(
            "evaluate",
            {
                "checkpoint": winner["checkpoint"],
                "policy": winner["policy"],
                "parameters": winner["inference_parameters"],
                "evaluation": final,
                "resources": resources,
            },
            root / "final",
        )
        save(final_path, {**result, "selection": winner, "source": source_identity()})
    return winner


# Preserve existing callers and saved launcher commands.
run_study = run_case


def check_manifest(specs, output=None):
    """Validate every case without Ray or training; return (rows, errors).

    With ``output``, also flag cases whose stored protocol.json a resume would
    refuse. Each row is (case, training trials, policies, inference trials, NFE,
    state) for a one-line summary per case.
    """
    rows, errors, names = [], [], set()
    for spec in specs:
        name = spec.get("name", "?")
        try:
            if name in names:
                raise ValueError("Duplicate case name")
            names.add(name)
            validate(spec)
            if not Path(spec["config"]).exists():
                raise FileNotFoundError(f"config {spec['config']} is missing")
            state = "new"
            if output is not None:
                stored = Path(output) / name / "protocol.json"
                if protocol_conflict(case_protocol(spec), stored):
                    raise ValueError(f"{stored} differs; a resume would refuse it")
                if stored.exists():
                    state = "resumes"
        except Exception as error:  # report every bad case, not only the first
            errors.append(f"{name}: {error}")
            continue
        policies = spec["inference"]
        rows.append((
            name,
            spec["training"]["trials"],
            len(policies),
            sum(policy["trials"] for policy in policies),
            policies[0].get("nfe", spec["evaluation"].get("nfe")),
            state,
        ))
    return rows, errors


def print_check(rows, errors):
    header = ("case", "train", "policies", "inference", "nfe", "state")
    width = max([len(header[0])] + [len(row[0]) for row in rows])
    print(f"{header[0]:{width}}  " + "  ".join(f"{h:>9}" for h in header[1:]))
    for row in rows:
        print(f"{row[0]:{width}}  " + "  ".join(f"{str(v):>9}" for v in row[1:]))
    trained = sum(row[1] for row in rows)
    print(f"{len(rows)} cases, {trained} training runs, "
          f"{sum(row[1] * row[3] for row in rows)} inference trials")
    for error in errors:
        print("ERROR", error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the manifest (and, with --output, stored protocols); run nothing",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--address", default="auto")
    parser.add_argument(
        "--working-dir", type=Path, help="Source directory shipped to Ray workers"
    )
    parser.add_argument(
        "--case-concurrency",
        "--study-concurrency",
        dest="case_concurrency",
        type=int,
        default=1,
        help="Independent experiment cases to execute concurrently",
    )
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="Retry candidates skipped after a failure (on_failure=skip) once more",
    )
    args = parser.parse_args()
    if args.check:
        rows, errors = check_manifest(json.loads(args.manifest.read_text()), args.output)
        print_check(rows, errors)
        raise SystemExit(1 if errors else 0)
    if args.output is None:
        parser.error("--output is required")
    if args.concurrency < 1 or args.case_concurrency < 1:
        parser.error("Concurrency must be positive")
    import ray

    options = {}
    if args.working_dir:
        options["runtime_env"] = {"working_dir": str(args.working_dir.resolve())}
    ray.init(address=args.address, **options)
    with ThreadPoolExecutor(max_workers=args.case_concurrency) as pool:
        futures = [
            pool.submit(
                run_case, spec, args.output, args.concurrency, args.retry_skipped
            )
            for spec in json.loads(args.manifest.read_text())
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
