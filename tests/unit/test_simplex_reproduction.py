from argparse import Namespace
from pathlib import Path
import pytest

from hydra import compose, initialize_config_dir

import main  # noqa: F401
from scripts.simplex.reproduce import experiment_matrix, commands


@pytest.mark.parametrize("weights", ["ema", "current"])
def test_full_manifest_budgets_and_hydra_commands(tmp_path, weights):
    args = Namespace(
        output=tmp_path, project="simplex-test", entity=None, weights=weights
    )
    entries = list(experiment_matrix([1, 2, 3, 4, 5], full=True))
    assert len(entries) == 160
    assert len({entry["name"] for entry in entries}) == 160
    jobs = [commands(entry, args) for entry in entries]
    assert sum(len(job["evaluations"]) for job in jobs) == 1050
    selected = [entries[0], entries[10], entries[100], entries[101], entries[102]]
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
        version_base=None,
    ):
        for entry in selected:
            job = commands(entry, args)
            for command in (job["train"], job["evaluations"][-1]):
                config = compose(config_name="config", overrides=command[2:])
                assert config.require_wandb and config.wandb.mode == "online"
                assert config.checkpointing.save_final
                assert config.eval.disable_ema == (weights == "current")
                assert f"weights-{weights}" in config.wandb.tags
                assert config.data.num_train == 200000 and config.data.num_valid == 5000
                assert config.trainer.max_steps == (
                    38462 if config.self_conditioning.enabled else 50000
                )
                assert config.model.full_precision and config.model.length == 180
                assert config.trainer.check_val_every_n_epoch is None
                assert (
                    config.trainer.limit_train_batches
                    % config.trainer.accumulate_grad_batches
                    == 0
                )
                if config.mode == "sudoku_eval":
                    assert config.eval.checkpoint_path.endswith("/final.ckpt")
