import json
import pytest

from core.sweeps import compose
from core.training.runner import run


@pytest.mark.parametrize("interval", [1, 2])
def test_validation_and_generation_epoch_interval(tmp_path, interval):
    cfg = compose(
        "smoke",
        {"output_dir": str(tmp_path / "run"), "data.cache_dir": str(tmp_path / "data")},
    )
    cfg.loader.global_batch_size = 4
    cfg.trainer.accumulate_grad_batches = 1
    cfg.trainer.max_steps = 4
    cfg.trainer.max_epochs = 2
    cfg.trainer.limit_train_batches = 1.0
    cfg.trainer.check_val_every_n_epoch = interval
    cfg.trainer.val_check_interval = 1.0
    cfg.trainer.enable_progress_bar = False
    cfg.callback.every_n_steps = 0
    cfg.callback.every_n_epochs = interval
    cfg.callback.evaluation_name = "validation"
    cfg.callback.save_records = False
    cfg.checkpoint.every_n_steps = 0
    cfg.checkpoint.every_n_epochs = 1
    cfg.checkpoint.save_top_k = 0
    run(cfg)
    files = sorted((tmp_path / "run/task_callback").glob("step_*.json"))
    records = [json.loads(p.read_text()) for p in files]
    assert len(records) == 2 // interval
    assert [r["epoch"] for r in records] == list(range(interval, 3, interval))
    assert all(r["evaluation_split"] == "validation" for r in records)
    assert all("records" not in r for r in records)
    assert (tmp_path / "run/checkpoints/last.ckpt").exists()
    assert (tmp_path / "run/checkpoints/best.ckpt").exists()
    import csv

    metrics = list(
        csv.DictReader(next((tmp_path / "run/metrics").glob("*/metrics.csv")).open())
    )
    assert sum(bool(row.get("val/nll")) for row in metrics) == 2 // interval
