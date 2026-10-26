import copy
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from core.sweeps.configuration import compose
from core.training.procedures.losses import SequenceAlpha, ReverseKL


def algorithm():
    cfg = compose(
        "corruption_factorial",
        overrides={
            "hybrid_view@model.views.content": "masked",
            "model.backbone.hidden_size": 32,
            "model.backbone.n_heads": 4,
            "model.backbone.n_blocks": 1,
            "model.backbone.cond_dim": 16,
            "model.backbone.dropout": 0.0,
            "self_conditioning.enabled": False,
        },
    )
    return instantiate(cfg.algorithm)


def test_alpha_ce_limit_and_shared_sequence_gradient():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    logp = logits.log_softmax(-1)
    tokens = torch.tensor([[1, 2, 3], [0, 1, 2]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    ce = SequenceAlpha(0)(logp, tokens, mask)
    expected = -logp[0, :2].gather(-1, tokens[0, :2, None]).mean()
    torch.testing.assert_close(ce[0], expected)
    assert ce[1] == 0
    alpha = SequenceAlpha(0.5)(logp, tokens, mask)
    a = torch.autograd.grad(alpha.sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad(ce.sum(), logits)[0]
    torch.testing.assert_close(a[0], b[0] * torch.exp(-0.5 * ce[0]))


@pytest.mark.parametrize("recipe", ["alpha", "sdtt"])
def test_configured_procedure_gradients_and_checkpoint(recipe):
    student = algorithm()
    cfg = OmegaConf.load(
        Path(__file__).parents[2] / "configs/procedure" / f"{recipe}.yaml"
    )
    procedure = instantiate(cfg, student=student)
    tokens = torch.randint(0, 9, (2, 81))
    batch = {
        "input_ids": tokens,
        "loss_mask": torch.ones_like(tokens, dtype=torch.bool),
    }
    result = procedure(student, batch, 0)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()
    )
    if procedure.teacher:
        assert all(p.grad is None for p in procedure.teacher.parameters())
        procedure.train()
        assert not procedure.teacher.training
        cloned = copy.deepcopy(procedure)
        cloned.load_state_dict(procedure.state_dict(), strict=True)
        assert all(
            torch.equal(v, cloned.state_dict()[k])
            for k, v in procedure.state_dict().items()
        )


def test_reverse_kl_zero_support():
    logp = torch.tensor([[[-0.693147, -0.693147, -float("inf")]]], requires_grad=True)
    value = ReverseKL()(logp, logp.detach(), None)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert torch.isfinite(logp.grad).all()


@pytest.mark.parametrize("recipe", ["dmmd", "idlm", "dmd"])
def test_alternating_roles_are_isolated(recipe):
    student = algorithm()
    if recipe == "dmd":
        cfg = compose(
            "corruption_factorial",
            overrides={
                "hybrid_view@model.views.content": "gaussian",
                "model.backbone.hidden_size": 32,
                "model.backbone.n_heads": 4,
                "model.backbone.n_blocks": 1,
                "model.backbone.cond_dim": 16,
                "model.backbone.dropout": 0.0,
                "self_conditioning.enabled": False,
            },
        )
        student = instantiate(cfg.algorithm)
    # Zero output initialization makes real/fake posterior identical; perturb
    # the auxiliary to make the generator gradient test nontrivial.
    config = OmegaConf.load(
        Path(__file__).parents[2] / "configs/procedure" / f"{recipe}.yaml"
    )
    procedure = instantiate(config, student=student)
    with torch.no_grad():
        procedure.auxiliary.model.backbone.output_layer.linear.bias.add_(
            torch.randn_like(
                procedure.auxiliary.model.backbone.output_layer.linear.bias
            )
            * 0.1
        )
    tokens = torch.randint(0, 9, (2, 81))
    batch = dict(input_ids=tokens, loss_mask=torch.ones_like(tokens, dtype=torch.bool))
    for i, phase in enumerate(procedure.phases):
        student.zero_grad(set_to_none=True)
        procedure.zero_grad(set_to_none=True)
        with procedure.active_role(student, phase.role):
            result = procedure(student, batch, 0, i)
            assert torch.isfinite(result["loss"])
            result["loss"].backward()
        active = student if phase.role == "student" else procedure.auxiliary
        inactive = student if phase.role == "auxiliary" else procedure.auxiliary
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in active.parameters()
        )
        assert all(p.grad is None for p in inactive.parameters())
        assert all(p.grad is None for p in procedure.teacher.parameters())


def test_manual_procedure_training_and_resume(tmp_path):
    import lightning as L
    from torch.utils.data import DataLoader
    from core.training.module import TrainingModule

    cfg = compose(
        "corruption_factorial",
        overrides={
            "hybrid_view@model.views.content": "masked",
            "model.backbone.hidden_size": 32,
            "model.backbone.n_heads": 4,
            "model.backbone.n_blocks": 1,
            "model.backbone.cond_dim": 16,
            "model.backbone.dropout": 0.0,
            "self_conditioning.enabled": False,
            "trainer.accumulate_grad_batches": 1,
            "trainer.gradient_clip_val": 0.0,
            "training.ema": 0.9,
            "training.warmup_steps": 0,
        },
    )
    cfg.procedure = OmegaConf.load("configs/procedure/dmmd.yaml")
    module = TrainingModule(cfg)
    batch = [
        dict(
            input_ids=torch.randint(0, 9, (81,)),
            loss_mask=torch.ones(81, dtype=torch.bool),
        )
        for _ in range(4)
    ]
    loader = DataLoader(batch, batch_size=2)
    trainer = L.Trainer(
        max_steps=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=loader)
    assert module.ema.num_updates == 1
    checkpoint = tmp_path / "last.ckpt"
    trainer.save_checkpoint(checkpoint)
    fresh = TrainingModule(cfg)
    trainer = L.Trainer(
        max_steps=4,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(fresh, train_dataloaders=loader, ckpt_path=str(checkpoint))
    assert fresh.ema.num_updates == 2


def test_promotion_state_survives_checkpoint():
    student = algorithm()
    cfg = OmegaConf.load("configs/procedure/sdtt.yaml")
    cfg.promotion_interval = 2
    procedure = instantiate(cfg, student=student)
    with torch.no_grad():
        next(student.parameters()).add_(0.1)
    batch = dict(
        input_ids=torch.zeros(2, 81, dtype=torch.long),
        loss_mask=torch.ones(2, 81, dtype=torch.bool),
    )
    procedure(student, batch, 2)
    assert int(procedure.last_promotion) == 2
    torch.testing.assert_close(
        next(procedure.teacher.parameters()), next(student.parameters())
    )
    torch.testing.assert_close(
        procedure.phases[0].targets.step_size, torch.tensor(1 / 64)
    )
    restored = instantiate(cfg, student=student)
    restored.load_state_dict(procedure.state_dict())
    restored(student, batch, 2)
    torch.testing.assert_close(
        restored.phases[0].targets.step_size, torch.tensor(1 / 64)
    )
