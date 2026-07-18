from types import MethodType, SimpleNamespace

import pytest
import torch
import lightning as L

from algos.trainer_base import LogLinear, TrainerBase, _unsqueeze
from models.ema import ExponentialMovingAverage


class TinyModule(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))


def _bare_trainer(*, ema=False, ignore_bos=False):
    trainer = object.__new__(TrainerBase)
    torch.nn.Module.__init__(trainer)
    trainer.backbone = TinyModule(2.0)
    trainer.noise = TinyModule(3.0)
    trainer.ignore_bos = ignore_bos
    trainer.ema = (
        ExponentialMovingAverage(trainer._get_parameters(), decay=0.9, use_num_updates=False)
        if ema
        else None
    )
    trainer.config = SimpleNamespace(loader=SimpleNamespace(eval_batch_size=2))
    return trainer


def test_log_linear_endpoints_and_derivative():
    schedule = LogLinear()
    t = torch.tensor([0.0, 0.25, 1.0])

    derivative, alpha = schedule(t)

    assert derivative == pytest.approx(-0.999)
    assert torch.allclose(alpha, torch.tensor([1.0, 0.75025, 0.001]))


@pytest.mark.parametrize(
    ("value_shape", "reference_shape", "expected_shape"),
    [((2,), (2, 3, 4), (2, 1, 1)), ((2, 3), (2, 3, 4), (2, 3, 1)), ((), (2, 3), (1, 1))],
)
def test_unsqueeze_appends_singleton_dimensions(value_shape, reference_shape, expected_shape):
    value = torch.zeros(value_shape)
    reference = torch.zeros(reference_shape)
    assert _unsqueeze(value, reference).shape == expected_shape


def test_loss_masks_invalid_tokens_and_aggregates():
    trainer = _bare_trainer()
    trainer._process_model_input = lambda x, valid: (x, x, valid)
    trainer.nll = lambda *_args, **_kwargs: torch.tensor([[1.0, 2.0, 100.0], [4.0, 5.0, 6.0]])
    valid = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32)

    result = trainer._loss(torch.zeros(2, 3, dtype=torch.long), valid)

    assert result.nlls.item() == pytest.approx(14.0)
    assert result.num_tokens.item() == pytest.approx(4.0)
    assert result.loss.item() == pytest.approx(3.5)
    assert result.prior_loss == 0.0


def test_loss_ignores_bos_without_mutating_attention_mask():
    trainer = _bare_trainer(ignore_bos=True)
    trainer._process_model_input = lambda x, valid: (x, x, valid)
    trainer.nll = lambda *_args, **_kwargs: torch.tensor([[100.0, 2.0, 3.0]])
    valid = torch.ones(1, 3)

    result = trainer._loss(torch.zeros(1, 3, dtype=torch.long), valid)

    assert result.nlls.item() == pytest.approx(5.0)
    assert result.num_tokens.item() == pytest.approx(2.0)
    assert torch.equal(valid, torch.ones_like(valid))


def test_loss_rejects_invalid_nll_shape():
    trainer = _bare_trainer()
    trainer._process_model_input = lambda x, valid: (x, x, valid)
    trainer.nll = lambda *_args, **_kwargs: torch.ones(2)

    with pytest.raises(AssertionError):
        trainer._loss(torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3))


def test_training_step_passes_accumulation_remainder(monkeypatch):
    trainer = _bare_trainer()
    trainer._test_trainer = SimpleNamespace(accumulate_grad_batches=3)
    monkeypatch.setattr(
        TrainerBase, "trainer", property(lambda self: self._test_trainer)
    )
    trainer.metrics = SimpleNamespace(update_train=lambda *_args: None)
    trainer.log = lambda **_kwargs: None
    observed = []

    def fake_loss(self, _tokens, _mask, accumulation_step, train_mode):
        observed.append((accumulation_step, train_mode))
        return SimpleNamespace(
            loss=torch.tensor(1.0), nlls=torch.tensor(1.0), prior_loss=0.0, num_tokens=torch.tensor(1.0)
        )

    trainer._loss = MethodType(fake_loss, trainer)
    batch = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2)}

    trainer.training_step(batch, batch_idx=5)

    assert observed == [(2, True)]


def test_eval_and_train_modes_swap_ema_and_module_modes():
    trainer = _bare_trainer(ema=True)
    original = [parameter.detach().clone() for parameter in trainer._get_parameters()]
    for shadow in trainer.ema.shadow_params:
        shadow.fill_(9.0)

    trainer._eval_mode()
    assert not trainer.backbone.training
    assert not trainer.noise.training
    assert all(parameter.item() == pytest.approx(9.0) for parameter in trainer._get_parameters())

    trainer._train_mode()
    assert trainer.backbone.training
    assert trainer.noise.training
    assert all(torch.equal(parameter, expected) for parameter, expected in zip(trainer._get_parameters(), original))


def test_checkpoint_load_cleans_compilation_and_teacher_keys():
    trainer = _bare_trainer()
    checkpoint = {
        "loops": {
            "fit_loop": {
                "epoch_progress": {"current": {"completed": 4}},
                "epoch_loop.batch_progress": {"current": {"completed": 7}},
            }
        },
        "state_dict": {
            "backbone._orig_mod.layer.weight": torch.tensor(1),
            "teacher.layer.weight": torch.tensor(2),
        },
    }

    trainer.on_load_checkpoint(checkpoint)

    assert trainer.fast_forward_epochs == 4
    assert trainer.fast_forward_batches == 7
    assert list(checkpoint["state_dict"]) == ["backbone.layer.weight"]


def test_restore_model_and_sample_restores_mode_and_ema():
    trainer = _bare_trainer(ema=True)
    trainer.config.loader.eval_batch_size = 2
    original = trainer.backbone.weight.detach().clone()
    trainer.ema.shadow_params[0].fill_(8.0)
    trainer.generate_samples = lambda **kwargs: torch.full((kwargs["num_samples"], 2), 5)

    samples = trainer.restore_model_and_sample(num_steps=3)

    assert samples.shape == (2, 2)
    assert trainer.backbone.training
    assert torch.equal(trainer.backbone.weight, original)


def test_restore_model_and_sample_restores_mode_after_exception():
    trainer = _bare_trainer(ema=True)
    original = trainer.backbone.weight.detach().clone()

    def fail(**_kwargs):
        raise RuntimeError("sampling failed")

    trainer.generate_samples = fail
    with pytest.raises(RuntimeError, match="sampling failed"):
        trainer.restore_model_and_sample(num_steps=3)

    assert trainer.backbone.training
    assert trainer.noise.training
    assert torch.equal(trainer.backbone.weight, original)


def test_configure_optimizers_wires_adamw_and_scheduler(monkeypatch):
    trainer = _bare_trainer()
    trainer.config.optim = SimpleNamespace(
        lr=0.02, beta1=0.8, beta2=0.95, eps=1e-7, weight_decay=0.03
    )
    trainer.config.lr_scheduler = object()
    scheduler = object()
    observed = {}

    def instantiate(config, optimizer):
        observed.update(config=config, optimizer=optimizer)
        return scheduler

    monkeypatch.setattr("algos.trainer_base.hydra.utils.instantiate", instantiate)
    optimizers, schedulers = trainer.configure_optimizers()
    optimizer = optimizers[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(0.02)
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["eps"] == pytest.approx(1e-7)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.03)
    assert observed == {"config": trainer.config.lr_scheduler, "optimizer": optimizer}
    assert schedulers == [{
        "scheduler": scheduler, "interval": "step", "monitor": "val/loss",
        "name": "trainer/lr",
    }]


def test_optimizer_step_updates_ema_once_after_optimizer(monkeypatch):
    trainer = _bare_trainer(ema=True)
    events = []
    monkeypatch.setattr(L.LightningModule, "optimizer_step", lambda *_a, **_k: events.append("step"))
    trainer.ema.update = lambda parameters: events.append(
        ("ema", len(list(parameters)))
    )
    trainer.optimizer_step("ignored")
    assert events == ["step", ("ema", 2)]


def _checkpoint_for_save(total_steps=5, current_steps=3):
    return {"loops": {"fit_loop": {
        "epoch_loop.batch_progress": {
            "total": {"completed": 0}, "current": {"completed": 0}
        },
        "epoch_loop.automatic_optimization.optim_progress": {
            "optimizer": {"step": {
                "total": {"completed": total_steps},
                "current": {"completed": current_steps},
            }}
        },
        "epoch_loop.state_dict": {"_batches_that_stepped": 0},
    }}}


def test_save_checkpoint_corrects_loop_counts_and_saves_sampler(monkeypatch):
    trainer = _bare_trainer(ema=True)
    random_state = torch.Generator().manual_seed(7).get_state()
    trainer._test_trainer = SimpleNamespace(
        accumulate_grad_batches=4,
        train_dataloader=SimpleNamespace(
            sampler=SimpleNamespace(state_dict=lambda: {"random_state": random_state})
        ),
    )
    monkeypatch.setattr(TrainerBase, "trainer", property(lambda self: self._test_trainer))
    checkpoint = _checkpoint_for_save()
    trainer.on_save_checkpoint(checkpoint)
    fit = checkpoint["loops"]["fit_loop"]
    assert fit["epoch_loop.batch_progress"]["total"]["completed"] == 20
    assert fit["epoch_loop.batch_progress"]["current"]["completed"] == 12
    assert fit["epoch_loop.state_dict"]["_batches_that_stepped"] == 5
    assert torch.equal(checkpoint["sampler"]["random_state"], random_state)
    assert "ema" in checkpoint


def test_train_start_replaces_loader_with_cpu_safe_sampler(monkeypatch):
    trainer = _bare_trainer()
    trainer.config.loader = SimpleNamespace(
        batch_size=2, num_workers=0, pin_memory=False
    )
    original = torch.utils.data.DataLoader(list(range(6)), batch_size=2, shuffle=True)
    combined = SimpleNamespace(flattened=[original])
    trainer._test_trainer = SimpleNamespace(
        _accelerator_connector=SimpleNamespace(
            use_distributed_sampler=False, is_distributed=False
        ),
        fit_loop=SimpleNamespace(_combined_loader=combined),
    )
    monkeypatch.setattr(TrainerBase, "trainer", property(lambda self: self._test_trainer))
    trainer.on_train_start()
    updated = combined.flattened[0]
    assert isinstance(updated.sampler, __import__("dataloader").RandomFaultTolerantSampler)
    assert updated.num_workers == 0
    assert not updated.persistent_workers


def test_validation_epoch_end_restores_training_mode_after_exception(monkeypatch):
    trainer = _bare_trainer(ema=True)
    trainer._eval_mode()
    trainer.metrics = SimpleNamespace(valid_nlls={})
    trainer.config.eval = SimpleNamespace(
        compute_perplexity_on_sanity=True,
        generate_samples=True,
        compute_generative_perplexity=False,
    )
    trainer.config.sampling = SimpleNamespace(num_sample_batches=1)
    trainer.generate_samples = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("validation sampling failed")
    )
    trainer._test_trainer = SimpleNamespace(sanity_checking=False)
    monkeypatch.setattr(TrainerBase, "trainer", property(lambda self: self._test_trainer))
    with pytest.raises(RuntimeError, match="validation sampling failed"):
        trainer.on_validation_epoch_end()
    assert trainer.backbone.training
    assert trainer.noise.training
