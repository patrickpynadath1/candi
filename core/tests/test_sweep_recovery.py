import json
from dataclasses import asdict


from core.sweeps.jobs import TrainingJob, InferenceJob
from core.sweeps.queue import _training_job
from core.sweeps.submissions import SubmissionQueue
from core.tests.test_programmatic_sweeps import tiny
from examples.recover_failed_sweep import recover


def test_recovery_preserves_checkpoint_and_evaluations_and_is_idempotent(tmp_path):
    queue = SubmissionQueue.create(tmp_path / "queue.sqlite")
    config = tiny()
    config["benchmark"] = {"algorithm_id": "original"}
    job = TrainingJob(
        name="trained",
        config=config,
        inference=(InferenceJob(overrides={"sampler.steps": 2}),),
    )
    queue.submit([job])
    directory = tmp_path / "run_job_abc"
    checkpoint = directory / "run/checkpoints/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (directory / "error.txt").write_text(
        "ValueError: could not convert string to float: 'False'"
    )
    trial = {"trial_id": "abc", "status": "ERROR", "config": asdict(_training_job(job))}
    (tmp_path / "experiment_state.json").write_text(
        json.dumps({"trial_data": [[json.dumps(trial), {}]]})
    )
    assert recover(tmp_path, queue)
    assert recover(tmp_path, queue)
    rows, _ = queue.read_after(0)
    assert len(rows) == 2
    saved = rows[1][1]
    assert saved["trial"]["config"]["checkpoint"]["resume"] == str(checkpoint)
    assert saved["trial"]["config"]["benchmark"]["algorithm_id"] == "original"
    assert saved["inference"][0]["overrides"]["sampler.steps"] == 2
    checkpoint.unlink()
    assert recover(tmp_path, queue)  # Accepted jobs no longer depend on old artifacts.


def test_recovery_waits_for_error_and_checkpoint_publication(tmp_path):
    queue = SubmissionQueue.create(tmp_path / "queue.sqlite")
    config = tiny()
    config["benchmark"] = {"algorithm_id": "original"}
    job = TrainingJob(name="delayed", config=config)
    queue.submit([job])
    assert not recover(tmp_path, queue)
    state = tmp_path / "experiment_state.json"
    state.write_text("{")
    assert not recover(tmp_path, queue)
    trial = {"trial_id": "abc", "status": "ERROR", "config": asdict(_training_job(job))}
    state.write_text(json.dumps({"trial_data": [[json.dumps(trial), {}]]}))
    assert not recover(tmp_path, queue)
    directory = tmp_path / "run_job_abc"
    directory.mkdir()
    (directory / "error.txt").touch()
    assert not recover(tmp_path, queue)
    (directory / "error.txt").write_text(
        "ValueError: could not convert string to float: 'False'"
    )
    assert not recover(tmp_path, queue)
    assert len(queue.read_after(0)[0]) == 1
    checkpoint = directory / "run/checkpoints/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    assert recover(tmp_path, queue)
    assert len(queue.read_after(0)[0]) == 2
    assert recover(tmp_path, queue)
    assert len(queue.read_after(0)[0]) == 2
