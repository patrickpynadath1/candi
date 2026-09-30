"""Tracking failures and checkpoint policy are independent experiment concerns."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

import main


@pytest.mark.parametrize("failure", [None, "initialization", "setup"])
def test_tracking_finalizes_setup_failures(monkeypatch, failure):
    events = []

    class Logger:
        @property
        def experiment(self):
            events.append("connect")
            if failure == "initialization":
                raise RuntimeError("tracking unavailable")
            return object()

        def finalize(self, status):
            events.append(status)

    monkeypatch.setattr(main.L.pytorch.loggers, "WandbLogger", lambda **kw: Logger())
    config = OmegaConf.create({"wandb": {"project": "test"}, "require_wandb": True})

    def run():
        with main._wandb_experiment(config):
            events.append("setup")
            if failure == "setup":
                raise RuntimeError("data unavailable")

    if failure:
        with pytest.raises(RuntimeError):
            run()
        assert events[-1] == "failed"
    else:
        run()
        assert events == ["connect", "setup", "success"]


@pytest.mark.parametrize("save_final", [False, True])
def test_final_checkpoint_does_not_require_tracking(monkeypatch, tmp_path, save_final):
    saved = []
    trainer = SimpleNamespace(
        fit=lambda *a, **kw: None,
        save_checkpoint=saved.append,
        is_global_zero=True,
        global_step=2,
    )
    config = OmegaConf.create(
        {
            "checkpointing": {
                "save_dir": str(tmp_path),
                "save_final": save_final,
                "resume_from_ckpt": False,
            },
            "data": {},
            "training": {"finetune_path": ""},
            "trainer": {},
            "strategy": {},
            "require_wandb": False,
        }
    )
    monkeypatch.setattr(main.dataloader, "get_dataloaders", lambda *a: (None, None))
    monkeypatch.setattr(main, "_print_batch", lambda *a, **kw: None)
    monkeypatch.setattr(main.hydra.utils, "instantiate", lambda *a, **kw: trainer)
    main._fit(lambda *a, **kw: object(), config, None, None)
    assert saved == ([str(tmp_path / "checkpoints/final.ckpt")] if save_final else [])
