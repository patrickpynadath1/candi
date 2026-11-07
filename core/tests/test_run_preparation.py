"""Per-run config preparation for nested-tuning workers."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from core.sweeps import compose
from core.sweeps.configuration import prepare_run


def tiny():
    config = compose("smoke")
    config.task = {
        "_target_": "core.tests.test_entrypoint.TinyTask",
        "vocab_size": 5,
        "length": 8,
    }
    config.model.hidden_size = 16
    config.model.n_heads = 2
    config.trainer.max_steps = 1
    config.trainer.enable_progress_bar = False
    config.trainer.limit_val_batches = 0
    config.checkpoint.every_n_steps = 1
    config.loader.num_workers = 0
    return OmegaConf.to_container(config, resolve=False)


@pytest.mark.parametrize(
    "method", sorted(path.stem for path in Path("configs/experiment").glob("*.yaml"))
)
def test_all_presets_resolve(method, tmp_path):
    config = compose(method)
    if config.get("strategy") is not None or config.trainer.strategy not in ("auto", None):
        pytest.skip("multi-GPU preset; nested-tuning workers hold one GPU")
    cfg = prepare_run({"experiment": method}, tmp_path, {"GPU": 1})
    from hydra.utils import get_class

    algorithm = get_class(cfg.algorithm._target_)
    assert callable(algorithm.loss) and callable(algorithm.generate)
    assert "${" not in OmegaConf.to_yaml(cfg)
    assert cfg.output_dir == str(tmp_path)


def test_editable_config_and_run_isolation(tmp_path):
    cfg = compose("gaussian_diffusion_sudoku", {"sampler": "heun"})
    assert cfg.sampler._target_.endswith("HeunSampler")
    cfg.model.hidden_size = 96
    spec = {"config": OmegaConf.to_container(cfg, resolve=False)}
    first = prepare_run(spec, tmp_path / "first", {"GPU": 1})
    second = prepare_run(spec, tmp_path / "second", {"CPU": 1})
    assert first.algorithm.model.hidden_size == 96
    assert first.evaluation.device == "cuda:0"
    assert second.evaluation.device == "cpu"
    assert first.output_dir != second.output_dir
    assert cfg.output_dir not in (first.output_dir, second.output_dir)
    first.task.length = 32
    assert second.task.length != 32


def test_mapping_overrides():
    config = compose(
        "sudoku_hard",
        {
            "++model.input_adapter": {"_target_": "core.models.components.TokenInput"},
            "++example": [{"key": "value"}, 2],
        },
    )
    assert config.example[0].key == "value"


@pytest.mark.parametrize("gpus", [-1, 0.5, 2, True, float("nan"), float("inf")])
def test_only_zero_or_one_whole_gpu(gpus, tmp_path):
    with pytest.raises(ValueError, match="zero or one whole GPU"):
        prepare_run({"config": tiny()}, tmp_path, {"GPU": gpus})


def test_bad_config_rejected(tmp_path):
    with pytest.raises(ValueError, match="either"):
        prepare_run({"config": tiny(), "experiment": "smoke"}, tmp_path, {})
    with pytest.raises(ValueError, match="Unknown"):
        prepare_run({"seed": 2}, tmp_path, {})
    cfg = tiny()
    cfg["trainer"]["num_nodes"] = 2
    with pytest.raises(ValueError, match="one node"):
        prepare_run({"config": cfg}, tmp_path, {})
    cfg = tiny()
    cfg["trainer"]["strategy"] = "ddp"
    with pytest.raises(ValueError, match="strategy=auto"):
        prepare_run({"config": cfg}, tmp_path, {})


@pytest.mark.parametrize("gpus", [0, 1])
def test_gpu_reservation_controls_devices(gpus, tmp_path):
    source = tiny()
    cfg = prepare_run({"config": source}, tmp_path, {"GPU": gpus})
    assert cfg.trainer.devices == 1
    assert cfg.trainer.accelerator == ("gpu" if gpus else "cpu")
    assert (
        cfg.trainer.accumulate_grad_batches
        == source["trainer"]["accumulate_grad_batches"]
    )
    assert cfg.loader.global_batch_size == source["loader"]["global_batch_size"]
