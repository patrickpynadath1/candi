import pytest
import torch

from scripts.latest_training_checkpoint import latest_checkpoint


def save_checkpoint(path, step):
    torch.save({"global_step": step, "optimizer_states": [{"state": {}}]}, path)


def test_restart_advances_past_original_best_checkpoint(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    original = directory / "best.ckpt"
    latest = directory / "last.ckpt"
    save_checkpoint(original, 1250)
    save_checkpoint(latest, 5000)
    assert latest_checkpoint(tmp_path, original) == str(latest)


def test_selects_step_not_filename_and_skips_partial_write(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    save_checkpoint(directory / "last.ckpt", 5000)
    save_checkpoint(directory / "best.ckpt", 6250)
    (directory / "0-10000.ckpt").write_bytes(b"interrupted write")
    assert latest_checkpoint(tmp_path) == str(directory / "best.ckpt")


def test_missing_checkpoint_fails_instead_of_silently_starting_over(tmp_path):
    with pytest.raises(RuntimeError, match="No resumable checkpoint"):
        latest_checkpoint(tmp_path)
