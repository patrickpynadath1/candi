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
import re
import traceback
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from core.evaluation.reporting import metric_value
from core.sweeps.configuration import prepare_run


INTERPOLATION = re.compile(r"^\$\{([A-Za-z_][\w.]*)\}$")


def resolve_key(config, key):
    """The dotted key a write to ``key`` must target, following interpolated parents.

    Unresolved configs alias nodes (``algorithm.predictor: ${parameterization}``).
    ``OmegaConf.update`` on a child of an alias would replace the alias by a new,
    incomplete node; writing to the alias target keeps every recipe field and every
    other reference to it consistent.
    """
    parts = key.split(".")
    node = config
    for index, part in enumerate(parts[:-1]):
        if not isinstance(node, DictConfig) or part not in node:
            return key
        if OmegaConf.is_interpolation(node, part):
            target = INTERPOLATION.match(str(node._get_node(part)._value()))
            if target is None:
                raise ValueError(f"cannot write {key} through the interpolation at {part}")
            return resolve_key(config, ".".join([target.group(1), *parts[index + 1:]]))
        node = node[part]
    return key


def update(config, key, value, **options):
    """``OmegaConf.update`` that writes through interpolated parent nodes."""
    OmegaConf.update(config, resolve_key(config, key), value, **options)


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
EXECUTION_FIELDS = ("resources", "wandb", "priority")


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


def active(distribution, values):
    """A parameter with ``when: {key: value or [values]}`` exists only under that condition."""
    when = {} if isinstance(distribution, list) else distribution.get("when") or {}
    return all(
        values.get(key) in (expected if isinstance(expected, list) else [expected])
        for key, expected in when.items()
    )


def suggest(trial, space):
    values = {}
    for key, distribution in space.items():
        if not active(distribution, values):
            continue  # The sampler keeps its configured value for this parameter.
        if isinstance(distribution, list):
            values[key] = trial.suggest_categorical(key, distribution)
        elif distribution["type"] == "categorical":
            values[key] = trial.suggest_categorical(key, distribution["choices"])
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
    budgets = spec["nfe"] if isinstance(spec["nfe"], list) else [spec["nfe"]]
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(type(b) is not int or b < 2 for b in budgets)
        or spec["trials"] < 1
    ):
        raise ValueError("Invalid NFE or trial budget")
    if spec["direction"] not in ("minimize", "maximize"):
        raise ValueError("Invalid optimization direction")
    if spec["budget_parameter"] not in (None, "steps", "max_steps"):
        raise ValueError("Unsupported sampler budget parameter")
    seen = []
    for key, distribution in spec["space"].items():
        if key in {"sampler.steps", "sampler.max_steps"} or not (
            key.startswith("sampler.") or key in PREDICTOR_PARAMETERS
        ):
            raise ValueError(f"Not a tunable inference parameter: {key}")
        when = {} if isinstance(distribution, list) else distribution.get("when") or {}
        if any(condition not in seen for condition in when):
            raise ValueError(f"{key}: a `when` condition must name an earlier parameter")
        seen.append(key)
    if "two_stage" in spec:
        validate_two_stage(spec)
    extra = spec.get("extra_test_nfe", [])
    if any(type(b) is not int or b < 2 for b in extra) or len(set(extra)) != len(extra) or (
        set(extra) & set(budgets)
    ):
        raise ValueError("extra_test_nfe must list distinct NFE budgets not used for tuning")
    count = spec.get("extra_test_examples", EXTRA_TEST_EXAMPLES)
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("extra_test_examples must be a positive integer or null")
    seeds = spec.get("extra_test_seeds", spec["test_seeds"][:1])
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(spec["test_seeds"]):
        raise ValueError("extra_test_seeds must be distinct test seeds")


# Sampling-only predictor settings a policy may tune (never the training objective).
PREDICTOR_PARAMETERS = {
    "algorithm.predictor.temperature",
    "algorithm.predictor.commitment",
    "algorithm.predictor.top_p",
}


def validate_two_stage(spec):
    stages = spec["two_stage"]
    first, second = stages.get("stage1"), stages.get("stage2")
    if any(type(n) is not int or n < 1 for n in (first, second)):
        raise ValueError("Two-stage selection needs positive stage1 and stage2 prompt counts")
    if first + second > len(spec["tuning_indices"]):
        raise ValueError("stage1 + stage2 prompts exceed the tuning pool")
    if type(stages.get("finalists", 3)) is not int or stages.get("finalists", 3) < 1:
        raise ValueError("Two-stage selection needs at least one finalist")
    seeds = stages.get("stage2_seeds", [])
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) & (
        set(spec["seeds"]) | set(spec["test_seeds"])
    ):
        raise ValueError("Stage-2 seeds must be fresh: unique, unlike tuning and test seeds")
    bootstrap = stages.get("stage2_bootstrap")
    if bootstrap is not None and (type(bootstrap) is not int or bootstrap < 1):
        raise ValueError("stage2_bootstrap must be a positive replicate count")
    if bootstrap and not spec["score"].get("per_example"):
        raise ValueError("stage2_bootstrap needs score.per_example (a per-record path)")
    for override in stages.get("stage2_overrides", {}):
        if not override.startswith("task."):
            raise ValueError(f"Not a stage-2 evaluation override: {override}")


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
    for override in {**evaluation.get("overrides", {}), **evaluation.get("test_overrides", {})}:
        if not override.startswith(("task.", "loader.eval_batch_size")):
            raise ValueError(f"Not an evaluation override: {override}")
    if evaluation.get("test_score", {"kind": "metric"}).get("kind") != "metric":
        raise ValueError("test_score must be a scalar task metric")
    for score in (evaluation["score"], evaluation.get("test_score", {})):
        if not isinstance(score.get("pool_seeds", False), bool):
            raise ValueError("score.pool_seeds must be true or false")
    resources = spec.get("resources", {"cpu": 4, "gpu": 1})
    if resources.get("gpu", 1) not in (0, 1) or resources.get("cpu", 4) < 1:
        raise ValueError("Workers require positive CPUs and zero or one whole GPU")
    if not 0 <= resources.get("inference_gpu", resources.get("gpu", 1)) <= 1:
        raise ValueError("inference_gpu is a GPU share between 0 and 1")


# Optuna samplers a study may use. The default reproduces the TPE search used by
# existing protocols; options are passed to the Optuna constructor unchanged.
SEARCH_ALGORITHMS = {
    "tpe": "TPESampler",
    "random": "RandomSampler",
    "qmc": "QMCSampler",
    # Gaussian-process Bayesian optimisation (needs torch and scipy); best for small,
    # continuous spaces. Parameters outside its relative space fall back to random.
    "gp": "GPSampler",
}
FAILURE_POLICIES = ("skip", "stop")


class StopSearch(RuntimeError):
    """A failure under on_failure=stop; an enclosing search never skips it."""


def make_sampler(settings, completed_trials=0):
    """Build a study's Optuna sampler from its ``search`` block.

    ``search: {algorithm: tpe|random|qmc|gp, ...options}``; the seed comes from the
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
        expected = {k for k, d in settings["space"].items() if active(d, old.params)}
        if set(old.params) != expected or any(
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
        update(config, key, value, merge=False, force_add=False)
    for key, value in payload["validation"].get("overrides", {}).items():
        update(config, key, value, force_add=True)
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


def evaluation_config(config, evaluation, policy, parameters):
    """A case config with an evaluation's overrides, a policy's sampler and a trial's values.

    The sampler's step count is set from the NFE budget. Shared by the evaluate
    worker and the tests that check every policy composes for every case.
    """
    from hydra.utils import instantiate

    from .budget import sampler_steps

    for key, value in evaluation.get("overrides", {}).items():
        update(config, key, value, merge=False, force_add=True)
    config.sampler = OmegaConf.create(policy["sampler"])
    for key, value in parameters.items():
        # A tuned algorithm key that does not exist (e.g. a temperature on an
        # algorithm without a predictor) would make every trial identical.
        if key.startswith("algorithm.") and (
            OmegaConf.select(config, key, default="__missing__") == "__missing__"
        ):
            raise ValueError(f"Unknown inference parameter for this recipe: {key}")
        update(config, key, value, merge=False, force_add=True)
    # Resolve after trial overrides: correction count can itself be tuned.
    sampler = instantiate(config.sampler)
    if policy["budget_parameter"] is not None:
        config.sampler[policy["budget_parameter"]] = sampler_steps(
            config, sampler, evaluation["nfe"]
        )
    return config


def _evaluate_worker(payload, output):
    from core.training.runner import run

    evaluation = payload["evaluation"]
    config = evaluation_config(
        OmegaConf.load(payload["checkpoint"]["config"]),
        evaluation, payload["policy"], payload["parameters"],
    )
    config.mode = "evaluate"
    # A tuning trial generates for seconds; compiling the transformer blocks costs more
    # than it saves, and leaves the GPU idle meanwhile. Outputs match within tolerance.
    config.compile = False
    config.evaluation.checkpoint = payload["checkpoint"]["checkpoint"]
    config.evaluation.legacy = False
    config.evaluation.max_examples = None
    config.evaluator = {
        "_target_": "core.evaluation.generation.evaluate",
        "_partial_": True,
        "evaluation_name": evaluation.get("dataset", "validation"),
        "indices": evaluation["indices"],
    }
    scores, intervals, examples, results = [], [], [], []
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
        results.append(result)
        scores.append(metric_value(result, evaluation["score"]["metric"]))
        intervals.append(result.get(f"{evaluation['score']['metric']}_interval"))
        if evaluation.get("bootstrap") and evaluation["score"].get("per_example"):
            examples.append(per_example_scores(result, evaluation["score"], scores[-1]))
    record = {"score": sum(scores) / len(scores), "seed_scores": scores}
    if examples:
        record["score_interval"] = mean_bootstrap(examples, evaluation["bootstrap"])
    elif any(interval is not None for interval in intervals):
        # A metric's own bootstrap interval (e.g. SOL with task.sol.bootstrap).
        record["seed_intervals"] = intervals
        if len(intervals) == 1:
            record["score_interval"] = intervals[0]
    if evaluation["score"].get("pool_seeds") and len(results) > 1:
        # One sample of every seed's generations (e.g. MAUVE at 2 x the prompts).
        pooled = pooled_result(prepared, evaluation.get("dataset", "validation"), results)
        save(Path(output) / "pooled-results.json", pooled)
        record["seed_mean_score"] = record["score"]
        record["score"] = metric_value(pooled, evaluation["score"]["metric"])
    return record


def pooled_result(config, name, results):
    """Task metrics of all seeds' records scored together (``score.pool_seeds``)."""
    from dataclasses import asdict

    from hydra.utils import instantiate

    metrics = instantiate(config.task).create_metric_accumulator(name)
    metrics.load_records([r for result in results for r in result["records"]])
    pooled = asdict(metrics.compute())
    pooled.pop("records", None)  # already saved per seed
    return pooled


def per_example_scores(result, score, aggregate):
    """{example index: value} at ``score.per_example`` in each record; checks the mean."""
    values = {
        int(record["index"]): metric_value(record, score["per_example"])
        for record in result["records"]
    }
    mean = sum(values.values()) / len(values)
    if not math.isclose(mean, aggregate, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"per_example {score['per_example']} averages to {mean}, not the metric {aggregate}"
        )
    return values


def mean_bootstrap(examples, replicates, confidence=0.95, seed=0):
    """Percentile interval of the seed-averaged mean, resampling examples (paired)."""
    import numpy as np

    keys = sorted(set.intersection(*(set(e) for e in examples)))
    values = np.array([[e[k] for k in keys] for e in examples], dtype=np.float64).mean(0)
    generator = np.random.default_rng(seed)
    means = [values[generator.integers(0, len(keys), len(keys))].mean() for _ in range(int(replicates))]
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return {"low": float(low), "high": float(high), "replicates": int(replicates),
            "confidence": confidence, "resampling": "paired_examples"}


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
    # Training needs a whole GPU; evaluation may share one (``inference_gpu``), since
    # short sampling trials leave most of a GPU idle.
    gpu = resources.get("gpu", 1)
    if stage != "train":
        gpu = resources.get("inference_gpu", gpu)
    remote = ray.remote(
        num_cpus=resources.get("cpu", 4),
        num_gpus=gpu,
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
        if k not in ("test_indices", "test_seeds", "test_dataset", "tuning_indices",
                     "test_score", "test_overrides")
    }
    validation["indices"] = evaluation["tuning_indices"]

    budgets = nfe_budgets(evaluation)

    def budget_directory(base, budget):
        # The first budget keeps the single-budget layout (status/analysis read it).
        return base if budget == budgets[0] else base.parent / f"{base.name}-nfe-{budget}"

    def inference(checkpoint, directory):
        winners = {}
        for budget in budgets:
            tuned = {**validation, "nfe": budget}
            choices = []
            for policy in spec["inference"]:

                def evaluate(parameters, output, indices=None, seeds=None, overrides=None,
                             bootstrap=None, policy=policy, tuned=tuned):
                    settings = dict(tuned)
                    if bootstrap:
                        settings["bootstrap"] = bootstrap
                    if indices is not None:
                        settings["indices"] = indices
                    if seeds is not None:
                        settings["seeds"] = seeds
                    if overrides:
                        settings["overrides"] = {**settings.get("overrides", {}), **overrides}
                    result = execute(
                        "evaluate",
                        {
                            "checkpoint": checkpoint,
                            "policy": policy,
                            "parameters": parameters,
                            "evaluation": settings,
                            "resources": resources,
                        },
                        output,
                    )
                    record = {
                        **result,
                        "checkpoint": checkpoint,
                        "policy": policy,
                        "inference_parameters": parameters,
                    }
                    if "two_stage" in evaluation or len(budgets) > 1:
                        record.update(
                            nfe=budget,
                            prompts=len(settings["indices"]),
                            seeds=list(settings["seeds"]),
                        )
                    return record

                folder = budget_directory(directory, budget) / policy["name"]
                if "two_stage" in evaluation:
                    choice = two_stage_search(
                        folder, policy, direction, evaluate, evaluation, concurrency,
                        retry_skipped,
                    )
                else:
                    choice = search(
                        folder, policy, direction, evaluate, concurrency, retry_skipped
                    )
                choices.append(choice)
            # A policy whose every trial failed (and was skipped) has no candidate.
            choices = [choice for choice in choices if choice is not None]
            if not choices:
                raise RuntimeError(
                    f"Every inference policy failed for {checkpoint['checkpoint']}"
                )
            choose = max if direction == "maximize" else min
            winners[budget] = choose(choices, key=lambda r: r["score"])
            if budget == budgets[0] and evaluation.get("extra_test_nfe"):
                # Each policy's own validation winner, for the extra test measurements.
                winners["policies"] = {c["policy"]["name"]: c for c in choices}
        first = winners[budgets[0]]
        if "policies" in winners:
            first = {**first, "policy_winners": winners["policies"]}
        if len(budgets) == 1:
            return first
        return {**first, "budgets": {str(b): winners[b] for b in budgets}}

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
    # With several NFE budgets each budget's winner is scored once on the test split.
    for budget in budgets:
        selection = winner if len(budgets) == 1 else winner["budgets"][str(budget)]
        final_dir = budget_directory(root / "final", budget)
        final_path = final_dir / "result.json"
        if final_path.exists():
            continue
        verify_checkpoint(selection["checkpoint"])
        final = {
            **validation,
            "nfe": budget,
            "dataset": evaluation.get(
                "test_dataset", validation.get("dataset", "validation")
            ),
            "indices": evaluation["test_indices"],
            "seeds": evaluation["test_seeds"],
        }
        final.pop("two_stage", None)
        final.update(test_settings(evaluation, final))
        result = execute(
            "evaluate",
            {
                "checkpoint": selection["checkpoint"],
                "policy": selection["policy"],
                "parameters": selection["inference_parameters"],
                "evaluation": final,
                "resources": resources,
            },
            final_dir,
        )
        save(final_path, {**result, "selection": selection, "source": source_identity()})
    # Extra measurements after selection, never read by it: every policy's validation
    # winner scored once on a fixed test subset at each extra NFE budget.
    extra_indices, extra_seeds = extra_test_subset(evaluation)
    for budget in evaluation.get("extra_test_nfe", []):
        for name, selection in winner["policy_winners"].items():
            folder = root / "final-extra" / f"nfe-{budget}" / name
            if (folder / "result.json").exists():
                continue
            verify_checkpoint(selection["checkpoint"])
            final = {
                **validation,
                "nfe": budget,
                "dataset": evaluation.get(
                    "test_dataset", validation.get("dataset", "validation")
                ),
                "indices": extra_indices,
                "seeds": extra_seeds,
            }
            final.pop("two_stage", None)
            final.update(test_settings(evaluation, final))
            result = execute(
                "evaluate",
                {
                    "checkpoint": selection["checkpoint"],
                    "policy": selection["policy"],
                    "parameters": selection["inference_parameters"],
                    "evaluation": final,
                    "resources": resources,
                },
                folder,
            )
            save(folder / "result.json", {
                **result, "nfe": budget, "measurement": "extra_test_nfe",
                "selection": selection, "source": source_identity(),
            })
    return winner


EXTRA_TEST_EXAMPLES = 128


def extra_test_subset(evaluation):
    """(indices, seeds) of the extra-budget smoke measurement: a fixed test subset.

    ``extra_test_examples`` prompts (default 128; null = the whole test partition) drawn
    with a fixed seed from ``test_indices``, and ``extra_test_seeds`` (default: the first
    test seed). The same subset for every policy and every resume.
    """
    import random

    test = list(evaluation["test_indices"])
    count = evaluation.get("extra_test_examples", EXTRA_TEST_EXAMPLES)
    indices = test if count is None or count >= len(test) else sorted(
        random.Random(0).sample(test, count)
    )
    return indices, list(evaluation.get("extra_test_seeds", evaluation["test_seeds"][:1]))


def test_settings(evaluation, final):
    """The test split's score and task overrides (``test_score``, ``test_overrides``).

    Tuning reads ``score``; a test evaluation may report another metric (SOL on
    TinyStories while ref_w1 tunes). Selection never reads test results.
    """
    settings = {}
    if evaluation.get("test_score"):
        settings["score"] = evaluation["test_score"]
    if evaluation.get("test_overrides"):
        settings["overrides"] = {**final.get("overrides", {}), **evaluation["test_overrides"]}
    return settings


def nfe_budgets(evaluation):
    """The NFE budgets a case tunes at: ``nfe`` is one integer or a list of them."""
    nfe = evaluation["nfe"]
    return list(nfe) if isinstance(nfe, list) else [nfe]


def ranked(entries, direction):
    """Best first; ties keep the earlier trial (deterministic on resume)."""
    sign = -1 if direction == "maximize" else 1
    return sorted(entries, key=lambda e: (sign * e["score"], e["trial"]))


def completed_trials(directory):
    """(trial number, parameters, score) of every completed trial of a study."""
    import optuna

    study = optuna.load_study(
        storage=f"sqlite:///{Path(directory).resolve() / 'optuna.sqlite'}", study_name="search"
    )
    return [
        {"trial": t.number, "parameters": t.params, "score": t.value}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]


def two_stage_subsets(evaluation):
    """(stage-1 indices, stage-2 indices): disjoint consecutive slices of the tuning pool."""
    stages = evaluation["two_stage"]
    pool, first = evaluation["tuning_indices"], stages["stage1"]
    return pool[:first], pool[first : first + stages["stage2"]]


def two_stage_search(directory, policy, direction, evaluate, evaluation, concurrency,
                     retry_skipped=False):
    """Two-stage selection of one policy's trials.

    Stage 1: the policy's Optuna study proposes ``trials`` candidates, each scored on the
    first ``stage1`` tuning prompts with the tuning seeds (the study sees only these
    scores). Stage 2: the best ``finalists`` (default 3) are re-scored on the next
    ``stage2`` prompts of the pool (disjoint from stage 1) with ``stage2_seeds`` and
    ``stage2_overrides`` (e.g. a bootstrap); the best stage-2 score is selected. Every
    evaluation is cached by its parameters, so a resumed run repeats nothing and
    selects the same candidate. Ties keep the earlier trial.
    """
    stages = evaluation["two_stage"]
    first_indices, second_indices = two_stage_subsets(evaluation)
    directory = Path(directory).resolve()
    skip = policy.get("on_failure", "skip") == "skip"
    log = logging.getLogger(__name__)

    def first(parameters, output):
        return evaluate(parameters, output, indices=first_indices)

    if search(directory, policy, direction, first, concurrency, retry_skipped) is None:
        return None
    entries = ranked(completed_trials(directory), direction)

    def score_round(name, candidates, **options):
        folder = directory / name

        def run(entry):
            key = hashlib.sha256(
                json.dumps(entry["parameters"], sort_keys=True).encode()
            ).hexdigest()[:20]
            output = folder / "candidates" / key
            path = output / "result.json"
            with lock(output / ".candidate.lock"):
                if not path.exists() and (output / "skipped.json").exists() and not retry_skipped:
                    return None  # A recorded skip stays skipped, as in search().
                if not path.exists():
                    try:
                        result = evaluate(entry["parameters"], output, **options)
                        if not math.isfinite(result["score"]):
                            raise ValueError("Nonfinite validation score")
                    except Exception as error:
                        save(output / "failure.json", {
                            "parameters": entry["parameters"],
                            "traceback": traceback.format_exc(),
                        })
                        if isinstance(error, StopSearch):
                            raise
                        if not skip:
                            raise StopSearch(f"{type(error).__name__}: {error}") from error
                        log.warning("Skipped %s candidate %s in %s: %s",
                                    name, entry["trial"], directory, error)
                        save(output / "skipped.json", {"error": f"{type(error).__name__}: {error}"})
                        return None
                    save(path, {**result, "source": source_identity()})
                result = json.loads(path.read_text())
            return {**entry, "score": result["score"], "result_path": str(path)}

        # As in search(): stop submitting after a failure, finish running work, then raise.
        scored, failure, queue = [], None, list(candidates)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = set()
            while queue or pending:
                while failure is None and queue and len(pending) < concurrency:
                    pending.add(pool.submit(run, queue.pop(0)))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                    except Exception as error:
                        failure = failure or error
                        continue
                    if result is not None:
                        scored.append(result)
        if failure is not None:
            raise failure
        return ranked(scored, direction)

    finalists = entries[: stages.get("finalists", 3)]
    final = score_round(
        "stage2",
        finalists,
        indices=second_indices,
        seeds=stages["stage2_seeds"],
        overrides=stages.get("stage2_overrides"),
        bootstrap=stages.get("stage2_bootstrap"),
    )
    if not final:
        return None
    best = final[0]
    result = json.loads(Path(best["result_path"]).read_text())
    selected = {
        **result,
        "parameters": best["parameters"],
        "two_stage": {"stage1": entries, "stage2": final},
    }
    save(directory / "selected.json", selected)
    return selected


# Preserve existing callers and saved launcher commands.
run_study = run_case


def generation_cost(spec):
    """Upper-bound generation work of one case: sequences and network calls.

    Counts every validation trial (both two-stage rounds) and the
    final test evaluation(s), per NFE budget and training candidate. Skipped
    candidates and early-stopping samplers make the real cost lower.
    """
    evaluation = spec["evaluation"]
    seeds = len(evaluation["seeds"])
    tuning = test = tuning_calls = test_calls = 0
    for budget in nfe_budgets(evaluation):
        sequences = 0
        for policy in spec["inference"]:
            trials = policy["trials"]
            if "two_stage" in evaluation:
                stages = evaluation["two_stage"]
                sequences += trials * stages["stage1"] * seeds
                finalists = min(stages.get("finalists", 3), trials)
                sequences += finalists * stages["stage2"] * len(stages["stage2_seeds"])
            else:
                sequences += trials * len(evaluation["tuning_indices"]) * seeds
        sequences *= spec["training"]["trials"]
        final = len(evaluation["test_indices"]) * len(evaluation["test_seeds"])
        tuning, test = tuning + sequences, test + final
        tuning_calls += sequences * budget
        test_calls += final * budget
    indices, extra_seeds = extra_test_subset(evaluation)
    final = len(indices) * len(extra_seeds)
    extra = sum(len(spec["inference"]) * final for _ in evaluation.get("extra_test_nfe", []))
    extra_calls = sum(
        len(spec["inference"]) * final * b for b in evaluation.get("extra_test_nfe", [])
    )
    return {
        "tuning_sequences": tuning, "test_sequences": test, "extra_test_sequences": extra,
        "tuning_calls": tuning_calls, "test_calls": test_calls, "extra_test_calls": extra_calls,
        "network_calls": tuning_calls + test_calls + extra_calls,
    }


def check_manifest(specs, output=None):
    """Validate every case without Ray or training; return (rows, errors).

    With ``output``, also flag cases whose stored protocol.json a resume would
    refuse. Each row is (case, training trials, policies, inference trials, NFE,
    state, generated sequences, network calls) for a one-line summary per case.
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
            checkpoint = spec["training"].get("checkpoint")
            if checkpoint and not Path(checkpoint).is_file():
                raise FileNotFoundError(f"fixed checkpoint {checkpoint} is missing")
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
        budgets = nfe_budgets(spec["evaluation"])
        rows.append((
            name,
            spec["training"]["trials"],
            len(policies),
            sum(policy["trials"] for policy in policies),
            budgets[0] if len(budgets) == 1 else "/".join(map(str, budgets)),
            state,
        ))
    return rows, errors


def print_check(rows, errors, costs=None):
    """One line per case; ``costs`` ({case: generation_cost}) adds two columns."""
    costs = costs or {}
    header = ("case", "train", "policies", "inference", "nfe", "state")
    if costs:
        header += ("sequences", "calls")
    width = max([len(header[0])] + [len(row[0]) for row in rows])
    print(f"{header[0]:{width}}  " + "  ".join(f"{h:>9}" for h in header[1:]))
    for row in rows:
        extra = ()
        if costs:
            cost = costs[row[0]]
            extra = (cost["tuning_sequences"] + cost["test_sequences"]
                     + cost["extra_test_sequences"], cost["network_calls"])
        print(f"{row[0]:{width}}  " + "  ".join(f"{str(v):>9}" for v in (*row[1:], *extra)))
    trained = sum(row[1] for row in rows)
    summary = (f"{len(rows)} cases, {trained} training runs, "
               f"{sum(row[1] * row[3] for row in rows)} inference trials")
    if costs:
        sequences = sum(c["tuning_sequences"] + c["test_sequences"] + c["extra_test_sequences"]
                        for c in costs.values())
        calls = sum(c["network_calls"] for c in costs.values())
        summary += f", {sequences} generated sequences, {calls / 1e6:.1f}M sequence network calls"
    print(summary)
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
        specs = json.loads(args.manifest.read_text())
        rows, errors = check_manifest(specs, args.output)
        valid = {row[0] for row in rows}
        costs = {s["name"]: generation_cost(s) for s in specs if s.get("name") in valid}
        print_check(rows, errors, costs)
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
