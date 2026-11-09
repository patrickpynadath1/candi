"""Checkpoint retention: frequent saves keep a bounded set of files that still resume."""

import torch

from core.tests.test_text8 import settings_for
from core.training.runner import run


def test_top_k_one_keeps_only_newest_step_last_and_best(tmp_path):
    settings = settings_for(tmp_path)
    settings.output_dir = str(tmp_path / "train")
    settings.trainer.enable_progress_bar = False
    settings.trainer.max_steps = 6
    settings.trainer.val_check_interval = 2
    settings.callback.every_n_steps = 0
    settings.checkpoint.every_n_steps = 2
    settings.checkpoint.save_top_k = 1
    run(settings)
    directory = tmp_path / "train/checkpoints"
    # Three periodic saves happened; only the newest step file survives, plus last and best.
    assert sorted(p.name for p in directory.iterdir()) == [
        "best.ckpt", "last.ckpt", "step_00000006.ckpt"
    ]
    # last.ckpt is what a resumed nested-tuning candidate continues from.
    assert torch.load(directory / "last.ckpt", map_location="cpu", weights_only=False)["global_step"] == 6
