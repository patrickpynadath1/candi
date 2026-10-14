"""Validation-only Optuna searches with Ray workers and one final evaluation.

A fixed checkpoint is the inference-only case. An optional training search adds
an outer loop; each trained checkpoint receives the same inference search budget.
The driver owns both studies and holds no GPUs while waiting for child work.
"""

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from core.evaluation.reporting import metric_value
from core.sweeps.tune import _prepare


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
    tune_ids, test_ids = spec["tuning_indices"], spec["test_indices"]
    if not tune_ids or not test_ids or set(tune_ids) & set(test_ids):
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
    if spec["budget_parameter"] not in ("steps", "max_steps"):
        raise ValueError("Unsupported sampler budget parameter")
    for key in spec["space"]:
        if key in {"sampler.steps", "sampler.max_steps"} or not (
            key.startswith("sampler.") or key == "algorithm.predictor.temperature"
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


def search(directory, settings, direction, objective, concurrency):
    """The same validation-only loop for training and inference candidates."""
    import optuna

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        storage=f"sqlite:///{directory / 'optuna.sqlite'}",
        study_name="search",
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=settings.get("seed", 42)),
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not study.trials:
        for point in settings.get("initial_points", []):
            study.enqueue_trial(point)

    def candidate(trial):
        parameters = suggest(trial, settings["space"])
        # Retry a completed stage without repeating training or generation.
        key = hashlib.sha256(
            json.dumps(parameters, sort_keys=True).encode()
        ).hexdigest()[:20]
        output = directory / "candidates" / key
        result_path = output / "result.json"
        with lock(output / ".candidate.lock"):
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                result = objective(parameters, output)
                if not math.isfinite(result["score"]):
                    raise ValueError("Nonfinite validation score")
                save(result_path, result)
        trial.set_user_attr("result_path", str(result_path))
        return result["score"]

    # Finish interrupted candidates before proposing new training configurations.
    for old in study.trials:
        if old.state == optuna.trial.TrialState.RUNNING:
            study.tell(old.number, state=optuna.trial.TrialState.FAIL)
        if old.state not in (
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.FAIL,
        ):
            continue
        if set(old.params) != set(settings["space"]) or any(
            t.params == old.params for t in completed
        ):
            continue
        retry = optuna.trial.FixedTrial(old.params)
        score = candidate(retry)
        recovered = optuna.trial.create_trial(
            params=old.params,
            distributions=old.distributions,
            value=score,
            user_attrs=retry.user_attrs,
        )
        study.add_trial(recovered)
        completed.append(recovered)

    remaining = settings["trials"] - len(completed)
    if remaining > 0:
        study.optimize(candidate, n_trials=remaining, n_jobs=concurrency)
    best = study.best_trial
    result = json.loads(Path(best.user_attrs["result_path"]).read_text())
    save(directory / "selected.json", {**result, "parameters": best.params})
    return {**result, "parameters": best.params}


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
    config = _prepare(
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
    if previous.exists():
        config.checkpoint.resume = str(previous)
    run(config)
    if not previous.is_file():
        raise FileNotFoundError(previous)
    evaluation_path = Path(output) / "evaluation-config.yaml"
    OmegaConf.save(OmegaConf.create(evaluation_config), evaluation_path)
    return {"config": str(evaluation_path), "checkpoint": str(previous)}


def _evaluate_worker(payload, output):
    from core.training.runner import run

    evaluation, policy = payload["evaluation"], payload["policy"]
    config = OmegaConf.load(payload["checkpoint"]["config"])
    for key, value in evaluation.get("overrides", {}).items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    config.sampler = OmegaConf.create(policy["sampler"])
    for key, value in payload["parameters"].items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    config.sampler[policy["budget_parameter"]] = evaluation["nfe"] - 1
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
        prepared = _prepare(
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


def execute(stage, payload, output):
    import ray

    worker = _train_worker if stage == "train" else _evaluate_worker
    resources = payload["resources"]
    remote = ray.remote(
        num_cpus=resources.get("cpu", 4), num_gpus=resources.get("gpu", 1), max_calls=1
    )(worker)
    return ray.get(remote.remote(payload, str(output)))


def run_study(spec, root, concurrency=4):
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    path = Path(root).resolve() / spec["name"] / ".driver.lock"
    with lock(path, blocking=False):
        return _run_study(spec, root, concurrency)


def _run_study(spec, root, concurrency):
    validate(spec)
    spec = copy.deepcopy(spec)
    root = Path(root).resolve() / spec["name"]
    root.mkdir(parents=True, exist_ok=True)
    # Pin the configuration contents, not just its mutable filename.
    config = OmegaConf.to_container(OmegaConf.load(spec["config"]), resolve=False)
    if "wandb" in spec:
        config["wandb"] = spec["wandb"]
    protocol = {"spec": spec, "config": config}
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a new study name")
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
        if k not in ("test_indices", "test_seeds", "tuning_indices")
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
                )
            )
        choose = max if direction == "maximize" else min
        return choose(choices, key=lambda r: r["score"])

    def train_candidate(parameters, output):
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output / "trained.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
        else:
            checkpoint = obtain_checkpoint(
                spec, config, parameters, validation, resources, output / "training"
            )
            save(checkpoint_path, checkpoint)
        result = inference(checkpoint, output / "inference")
        return {**result, "training_parameters": parameters}

    winner = search(root / "training", spec["training"], direction, train_candidate, 1)
    save(root / "selected.json", winner)
    # No final evaluation occurs until both levels and sampler selection finish.
    final_path = root / "final/result.json"
    if not final_path.exists():
        final = {
            **validation,
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
        save(final_path, {**result, "selection": winner})
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--address", default="auto")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("Concurrency must be positive")
    import ray

    ray.init(address=args.address)
    for spec in json.loads(args.manifest.read_text()):
        run_study(spec, args.output, args.concurrency)


if __name__ == "__main__":
    main()
