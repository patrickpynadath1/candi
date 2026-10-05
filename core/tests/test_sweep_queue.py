"""Dynamic dependencies and persistence on top of native Ray Tune execution."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from core.sweeps import InferenceJob, TrainingJob
from core.sweeps.queue import TrainingInferenceSearch, run_job
from core.tests.test_programmatic_sweeps import tiny


def jobs():
    return [
        TrainingJob(
            name=name,
            config=tiny(),
            inference=(
                InferenceJob(
                    sampler="ancestral",
                    overrides={"sampler.steps": 2, "evaluation.max_examples": 4},
                ),
                InferenceJob(
                    sampler="entropy_bounded", overrides={"evaluation.max_examples": 4}
                ),
            ),
        )
        for name in ["fast", "slow"]
    ]


def persisted_result(tmp_path):
    saved = tmp_path / "checkpoint_000000"
    (saved / "checkpoints").mkdir(parents=True)
    cfg = OmegaConf.create(tiny())
    OmegaConf.save(cfg, saved / "config.yaml")
    (saved / "checkpoints/last.ckpt").touch()
    return {
        "artifact_root": str(tmp_path),
        "artifact_filesystem": "local",
        "checkpoint_dir_name": saved.name,
    }


def test_dynamic_expansion_priority_pause_and_persistence(tmp_path):
    search = TrainingInferenceSearch(jobs())
    first = search.suggest("train-1")
    assert first["name"] == "fast"
    # Append descendants ahead of training not yet handed to Tune.
    result = persisted_result(tmp_path)
    search.on_trial_complete("train-1", result)
    checkpoint = tmp_path / "queue.json"
    search.save(checkpoint)
    restored = TrainingInferenceSearch([])
    restored.restore(checkpoint)
    restored.on_trial_complete(
        "train-1", result
    )  # Duplicate notification after restore.
    infer1 = restored.suggest("infer-1")
    infer2 = restored.suggest("infer-2")
    assert infer1["stage"] == infer2["stage"] == "inference"
    assert infer1["parent"] == infer2["parent"] == "train-1"
    assert (
        infer1["trial"]["config"]["evaluation"]["checkpoint"]
        == infer2["trial"]["config"]["evaluation"]["checkpoint"]
    )
    assert infer1["trial"]["config"]["model"] == first["trial"]["config"]["model"]
    assert restored.suggest("train-2")["name"] == "slow"
    assert restored.suggest("waiting") is None
    restored.on_trial_complete("infer-1", {})
    restored.on_trial_complete("infer-2", {})
    # An in-flight parent must keep the search open through another save/restore.
    restored.save(checkpoint)
    resumed = TrainingInferenceSearch([])
    resumed.restore(checkpoint)
    assert resumed.suggest("waiting") is None
    resumed.on_trial_complete("train-2", error=True)
    assert resumed.suggest("finished") == resumed.FINISHED


def test_failed_parent_does_not_expand_and_missing_artifacts_fail(tmp_path):
    search = TrainingInferenceSearch(jobs()[:1])
    search.suggest("parent")
    with pytest.raises(ValueError, match="persisted"):
        search.on_trial_complete("parent", {})
    search.on_trial_complete("parent", error=True)
    assert search.suggest("next") == search.FINISHED
    assert not search.expanded


@pytest.mark.parametrize(
    "change",
    [
        {"model.hidden_size": 32},
        {"evaluation.checkpoint": "other.ckpt"},
        {"parameterization.temperature": 0.5},
    ],
)
def test_inference_preserves_checkpoint_and_architecture(change):
    with pytest.raises(ValueError, match="cannot override"):
        InferenceJob(overrides=change)


def test_training_job_validation_and_queue_snapshot():
    with pytest.raises(ValueError, match="either"):
        TrainingJob(name="bad", experiment="smoke", config=tiny())
    with pytest.raises(ValueError, match="filename"):
        TrainingJob(name="bad", checkpoint="../last.ckpt")
    source = jobs()[0]
    search = TrainingInferenceSearch([source])
    source.config["seed"] = 99
    source.inference[0].overrides["sampler.steps"] = 99
    queued = search.suggest("train")
    assert queued["trial"]["config"]["seed"] != 99
    assert queued["inference"][0]["overrides"]["sampler.steps"] == 2


def test_real_tune_inference_starts_before_training_finishes(tmp_path, monkeypatch):
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray
    from ray import tune

    ray.init(
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024**2,
    )
    try:

        @ray.remote(num_cpus=0)
        class Events:
            def __init__(self):
                self.events = []

            def add(self, name, stage, event):
                self.events.append((name, stage, event))

            def get(self):
                return self.events

        events = Events.remote()

        def tracked(job):
            import time

            ray.get(events.add.remote(job["name"], job["stage"], "start"))
            if job["name"] == "slow":
                # A barrier, not an assumed training duration: the slow training
                # job can finish only after an inference worker has actually run.
                deadline = time.monotonic() + 60
                while not any(
                    stage == "inference" for _, stage, _ in ray.get(events.get.remote())
                ):
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "Inference waited for the entire training sweep"
                        )
                    time.sleep(0.1)
            run_job(job)
            ray.get(events.add.remote(job["name"], job["stage"], "finish"))

        trainable = tune.with_resources(tracked, {"cpu": 1})
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=TrainingInferenceSearch(jobs()),
                num_samples=-1,
                max_concurrent_trials=2,
            ),
            run_config=tune.RunConfig(
                storage_path=str(tmp_path), name="dynamic", verbose=0
            ),
        )
        results = tuner.fit()
        assert results.num_errors == 0, results.errors
        assert len(results) == 6
        history = ray.get(events.get.remote())
        first_inference = next(
            i
            for i, (_, stage, event) in enumerate(history)
            if stage == "inference" and event == "start"
        )
        slow_finished = history.index(("slow", "train", "finish"))
        assert first_inference < slow_finished
        parents = {
            result.metrics["trial_id"]
            for result in results
            if result.config["stage"] == "train"
        }
        for result in results:
            if result.config["stage"] == "inference":
                assert result.config["parent"] in parents
                assert result.metrics["num_total"] == 4
                assert Path(
                    result.config["trial"]["config"]["evaluation"]["checkpoint"]
                ).is_file()
        restored = tune.Tuner.restore(
            str(tmp_path / "dynamic"), trainable=trainable
        ).fit()
        assert len(restored) == 6
        assert ray.get(events.get.remote()) == history
    finally:
        ray.shutdown()
