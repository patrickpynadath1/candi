import math

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from core.tests.test_mdlm import config
from core.training.module import TrainingModule
from core.training.schedules import constant_warmup, cosine_decay_warmup, step_scheduler


def optimizer():
    return torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": 0.01},
            {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": 0.1},
        ]
    )


def rates(schedule, opt, steps):
    result = [schedule.get_last_lr()]
    for _ in range(steps):
        opt.step()
        schedule.step()
        result.append(schedule.get_last_lr())
    return result


def test_warmup_and_halving_match_legacy():
    for warmup in (0, 3):
        opt = optimizer()
        actual = rates(constant_warmup(opt, warmup), opt, 8)
        assert [r[0] for r in actual] == pytest.approx(
            [0.01 * (min(1.0, step / warmup) if warmup else 1.0) for step in range(9)]
        )
    opt = optimizer()
    actual = rates(step_scheduler(opt, 2, 3), opt, 9)
    assert [r[0] for r in actual] == pytest.approx(
        [0, 0.005, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.0025, 0.0025]
    )


def test_cosine_group_rates_and_resume():
    opt = optimizer()
    schedule = cosine_decay_warmup(
        opt, 10, warmup_steps=2, warmup_lr_init=0.001, lr_min=0.0001
    )
    first = rates(schedule, opt, 5)
    assert first[0] == pytest.approx([0.001, 0.001])
    assert first[2] == pytest.approx([0.01, 0.1])
    for i, peak in enumerate([0.01, 0.1]):
        assert first[5][i] == pytest.approx(
            0.0001 + (peak - 0.0001) * (1 + math.cos(math.pi * 3 / 8)) / 2
        )
    saved_opt, saved_schedule = opt.state_dict(), schedule.state_dict()
    remaining = rates(schedule, opt, 8)
    other = optimizer()
    restored = cosine_decay_warmup(
        other, 10, warmup_steps=2, warmup_lr_init=0.001, lr_min=0.0001
    )
    other.load_state_dict(saved_opt)
    restored.load_state_dict(saved_schedule)
    assert rates(restored, other, 8) == remaining
    assert remaining[-1] == pytest.approx([0.0001, 0.0001])


@pytest.mark.parametrize(
    "name", ["constant", "constant_warmup", "step_scheduler", "cosine_decay_warmup"]
)
def test_scheduler_config_construction(name):
    settings = config(f"lr_scheduler={name}")
    result = TrainingModule(settings).configure_optimizers()
    assert result["lr_scheduler"]["interval"] == "step"
    assert isinstance(result["optimizer"], torch.optim.AdamW)


def test_legacy_config_retains_schedule():
    settings = config()
    del settings.lr_scheduler
    result = TrainingModule(settings).configure_optimizers()
    assert result["lr_scheduler"]["scheduler"].get_last_lr() == [0.0]


def test_strategy_presets_do_not_replace_legacy_override():
    settings = config("trainer.strategy=ddp")
    assert settings.get("strategy") is None
    assert settings.trainer.strategy == "ddp"
    settings = config("strategy=ddp")
    assert instantiate(settings.strategy)._ddp_kwargs["find_unused_parameters"]
    settings = config("strategy=single_cpu")
    assert instantiate(settings.strategy).root_device == torch.device("cpu")
    assert (
        OmegaConf.to_container(settings.lr_scheduler)["warmup_steps"]
        == "${training.warmup_steps}"
    )


def test_fsdp_rejects_unsupported_lifecycle():
    from core.training.strategies import build_strategy

    settings = config("strategy=fsdp")
    algorithm = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="training.ema=0"):
        build_strategy(settings, algorithm)
    settings.training.ema = 0
    settings.callback.every_n_steps = 1
    with pytest.raises(ValueError, match="callback.every_n_steps=0"):
        build_strategy(settings, algorithm)
    settings.callback.every_n_steps = 0
    with pytest.raises(ValueError, match="gradient_clip_val=0"):
        build_strategy(settings, algorithm)
    settings.trainer.gradient_clip_val = 0
    assert type(build_strategy(settings, algorithm)).__name__ == "FSDPStrategy"
    algorithm.post_optimizer_step = lambda: None
    with pytest.raises(ValueError, match="projections"):
        build_strategy(settings, algorithm)


def test_legacy_tokenizer_metadata_is_inert():
    import io
    import pickle
    from core.training.legacy_pickle import Unpickler

    # A GLOBAL opcode referencing the removed Transformers 4 module.
    saved = (
        b"ctransformers.models.gpt2.tokenization_gpt2_fast\nGPT2TokenizerFast\n)\x81."
    )
    tokenizer = Unpickler(io.BytesIO(saved)).load()
    tokenizer.__setstate__({"vocab": "unused checkpoint metadata"})
    assert tokenizer.saved_state["vocab"] == "unused checkpoint metadata"
    assert not callable(tokenizer)
    assert pickle.loads(pickle.dumps(tokenizer)).saved_state == tokenizer.saved_state
