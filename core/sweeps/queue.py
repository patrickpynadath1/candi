"""A dependency-aware job source for Tune; Tune still schedules every worker."""

from collections import deque
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from omegaconf import OmegaConf
from ray.tune.search import Searcher

from core.sweeps.configuration import compose
from core.sweeps.jobs import InferenceJob, TrainingJob
from core.sweeps.tune import _configuration, run_trial


def run_job(job):
    """Tune trainable for configurations emitted by TrainingInferenceSearch."""
    run_trial(job["trial"])


@dataclass(frozen=True)
class _QueuedJob:
    stage: Literal["train", "inference"]
    name: str
    trial: dict[str, Any]
    parent: str | None = None
    inference: tuple[InferenceJob, ...] = ()
    checkpoint: str = "last.ckpt"

    @classmethod
    def from_dict(cls, values):
        return cls(
            **{
                **values,
                "inference": tuple(
                    InferenceJob(**choice) for choice in values.get("inference", [])
                ),
            }
        )


def _training_job(source):
    source = copy.deepcopy(source)
    config = _configuration(source.to_trial_spec())
    if config.mode != "train":
        raise ValueError("A training job must have mode=train")
    return _QueuedJob(
        stage="train",
        name=source.name,
        trial={"config": OmegaConf.to_container(config, resolve=False)},
        inference=tuple(
            InferenceJob(choice.sampler, choice.overrides)
            for choice in source.inference
        ),
        checkpoint=source.checkpoint,
    )


def _inference_config(saved: Path, choice: InferenceJob, checkpoint: str):
    config = OmegaConf.load(saved / "config.yaml")
    if choice.sampler is not None:
        sampler = compose(overrides={"sampler": choice.sampler}).sampler
        config.sampler = OmegaConf.to_container(sampler, resolve=False)
    for key, value in choice.overrides.items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    config.mode = "evaluate"
    config.evaluation.checkpoint = str(saved / "checkpoints" / checkpoint)
    config.evaluation.legacy = False
    if not Path(config.evaluation.checkpoint).is_file():
        raise FileNotFoundError(config.evaluation.checkpoint)
    return OmegaConf.to_container(config, resolve=False)


class TrainingInferenceSearch(Searcher):
    """Emit training jobs and expand successful checkpoints into inference jobs.

    Jobs are TrainingJob dataclasses with attached InferenceJob definitions.
    Dictionaries are used only at Tune and persistence boundaries.
    Use TuneConfig(search_alg=..., num_samples=-1, max_concurrent_trials=...).
    Checkpoint storage must be a local/shared filesystem visible to all workers.
    """

    def __init__(self, jobs: Iterable[TrainingJob] = (), *, submission_path=None):
        super().__init__()
        self.training: deque[_QueuedJob] = deque()
        self.inference: deque[_QueuedJob] = deque()
        self.issued: dict[str, _QueuedJob] = {}
        self.completed: dict[str, Literal["failed", "succeeded"]] = {}
        self.expanded: set[str] = set()
        self.submission_path = (
            str(Path(submission_path).resolve()) if submission_path else None
        )
        self.submission_cursor = 0
        self.names: set[str] = set()
        for source in jobs:
            job = _training_job(source)
            if job.name in self.names:
                raise ValueError(f"Duplicate job name: {job.name!r}")
            self.names.add(job.name)
            self.training.append(job)

    def _poll_submissions(self):
        if not self.submission_path:
            return False
        from core.sweeps.submissions import SubmissionQueue

        rows, opened = SubmissionQueue(self.submission_path).read_after(
            self.submission_cursor
        )
        for sequence, payload in rows:
            job = _QueuedJob.from_dict(payload)
            if job.name in self.names:
                raise ValueError(f"Duplicate job name: {job.name!r}")
            self.training.append(job)
            self.names.add(job.name)
            self.submission_cursor = sequence
        return opened

    def suggest(self, trial_id):
        if trial_id in self.issued:
            return asdict(self.issued[trial_id])
        opened = self._poll_submissions()
        queue = self.inference if self.inference else self.training
        if queue:
            job = queue.popleft()
            self.issued[trial_id] = job
            return asdict(job)
        # Empty while training is running is a pause, not end of the sweep.
        if opened or self.issued.keys() - self.completed.keys():
            return None
        return self.FINISHED

    def on_trial_complete(self, trial_id, result=None, error=False):
        if self.completed.get(trial_id) == "succeeded":
            return
        job = self.issued[trial_id]
        children = []
        if not error and job.stage == "train" and job.inference:
            if not result or not result.get("checkpoint_dir_name"):
                raise ValueError(
                    "Training finished without a persisted Tune checkpoint"
                )
            if result.get("artifact_filesystem") != "local":
                raise ValueError(
                    "Dynamic inference requires local/shared filesystem storage"
                )
            saved = Path(result["artifact_root"]) / result["checkpoint_dir_name"]
            if trial_id not in self.expanded:
                for index, choice in enumerate(job.inference):
                    config = _inference_config(saved, choice, job.checkpoint)
                    children.append(
                        _QueuedJob(
                            stage="inference",
                            name=f"{job.name}/inference-{index}",
                            parent=trial_id,
                            trial={"config": config},
                        )
                    )
                self.expanded.add(trial_id)
        self.inference.extend(children)
        self.completed[trial_id] = "failed" if error else "succeeded"

    def save(self, checkpoint_path):
        # Tune owns checkpoint publication; persist all queue bookkeeping here.
        Path(checkpoint_path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "submission_path": self.submission_path,
                    "submission_cursor": self.submission_cursor,
                    "training": [asdict(job) for job in self.training],
                    "inference": [asdict(job) for job in self.inference],
                    "issued": {
                        trial: asdict(job) for trial, job in self.issued.items()
                    },
                    "completed": self.completed,
                    "expanded": sorted(self.expanded),
                }
            )
        )

    def restore(self, checkpoint_path):
        state = json.loads(Path(checkpoint_path).read_text())
        if state["version"] != 1:
            raise ValueError("Unsupported training/inference queue version")
        self.training = deque(_QueuedJob.from_dict(job) for job in state["training"])
        self.inference = deque(_QueuedJob.from_dict(job) for job in state["inference"])
        self.issued = {
            trial: _QueuedJob.from_dict(job) for trial, job in state["issued"].items()
        }
        self.completed = state["completed"]
        self.expanded = set(state["expanded"])

        self.submission_path = state.get(
            "submission_path", getattr(self, "submission_path", None)
        )
        self.submission_cursor = state.get("submission_cursor", 0)
        self.names = {job.name for job in self.training}
        self.names.update(
            job.name for job in self.issued.values() if job.stage == "train"
        )
