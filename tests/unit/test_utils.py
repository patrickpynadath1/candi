from pathlib import Path
from types import SimpleNamespace
import io
import logging

import pytest
import torch

import utils


def test_lr_halve_scheduler_warmup_and_boundaries():
    scheduler = utils.LRHalveScheduler(warmup_steps=4, n_halve_steps=3)
    assert [scheduler(step) for step in range(4)] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert scheduler(4) == pytest.approx(1.0)
    assert scheduler(6) == pytest.approx(1.0)
    assert scheduler(7) == pytest.approx(0.5)
    assert scheduler(10) == pytest.approx(0.25)


def test_top_k_keeps_only_largest_values():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    filtered = utils.top_k_top_p_filtering(logits.clone(), top_k=2)
    assert torch.isfinite(filtered).tolist() == [[False, True, True, False]]


def test_top_p_keeps_minimal_nucleus_and_at_least_one_token():
    logits = torch.tensor([[10.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    filtered = utils.top_k_top_p_filtering(logits.clone(), top_p=0.01)
    assert torch.isfinite(filtered).sum(dim=-1).tolist() == [1, 1]
    assert filtered[0, 0] == 10.0


def test_top_k_top_p_combination_preserves_shape():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    filtered = utils.top_k_top_p_filtering(logits.clone(), top_k=3, top_p=0.7)
    assert filtered.shape == logits.shape
    assert torch.isfinite(filtered).any(dim=-1).all()
    assert torch.isneginf(filtered[0, 3])


def test_filtering_alternate_dimension():
    logits = torch.tensor([[1.0, 4.0], [3.0, 2.0], [2.0, 1.0]])
    filtered = utils.top_k_top_p_filtering(logits.clone(), top_k=2, dim=0)

    assert filtered.shape == logits.shape
    assert torch.isfinite(filtered).sum(dim=0).tolist() == [2, 2]
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[2, 1])


def test_fsspec_helpers_with_temporary_directory(tmp_path):
    nested = tmp_path / "a" / "b"
    utils.fsspec_mkdirs(str(nested))
    file_path = nested / "value.txt"
    file_path.write_text("content")

    assert utils.fsspec_exists(str(file_path))
    listing = utils.fsspec_listdir(str(nested))
    assert any(Path(entry).name == "value.txt" for entry in listing)


def test_count_parameters_ignores_frozen_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[1].requires_grad_(False)
    assert utils.count_parameters(model) == 9


def test_logging_context_restores_level_and_handler():
    logger = logging.getLogger("unit-test-logging-context")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(io.StringIO())
    with utils.LoggingContext(logger, level=logging.DEBUG, handler=handler, close=False):
        assert logger.level == logging.DEBUG
        assert handler in logger.handlers
    assert logger.level == logging.WARNING
    assert handler not in logger.handlers


def test_gradient_callback_respects_window_and_ignores_missing_gradients():
    callback = utils.GradientInspectionCallback(num_grads_log=2)
    block = torch.nn.Linear(2, 1)
    module = SimpleNamespace(
        backbone=SimpleNamespace(blocks=block), device=torch.device("cpu")
    )
    callback.on_before_optimizer_step(None, module, None)
    assert not hasattr(module, "grad_accum_buffer")
    block(torch.ones(1, 2)).sum().backward()
    callback.on_before_optimizer_step(None, module, None)
    assert callback.num_grads_log == 2
    assert module.grad_step == 1


def test_gradient_callback_rejects_invalid_window():
    with pytest.raises(ValueError, match="positive"):
        utils.GradientInspectionCallback(0)


def test_get_aux_model_loads_checkpoint_and_freezes_parameters(monkeypatch):
    checkpoint = {"hyper_parameters": {"config": {"algo": {"name": "tiny"}}}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    loaded = torch.nn.Linear(2, 2)
    calls = []

    class Algorithm:
        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            calls.append((path, kwargs))
            return loaded

    registry = SimpleNamespace(get=lambda name: Algorithm if name == "tiny" else None)
    result = utils.get_aux_model("checkpoint.ckpt", registry, tokenizer="tok")
    assert result is loaded
    assert not result.training
    assert all(not parameter.requires_grad for parameter in result.parameters())
    assert calls == [(
        "checkpoint.ckpt",
        {"config": checkpoint["hyper_parameters"]["config"], "tokenizer": "tok"},
    )]


def test_get_aux_model_can_preserve_gradients(monkeypatch):
    checkpoint = {"hyper_parameters": {"config": {"algo": {"name": "tiny"}}}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    loaded = torch.nn.Linear(1, 1)

    class Algorithm:
        load_from_checkpoint = staticmethod(lambda *args, **kwargs: loaded)

    registry = SimpleNamespace(get=lambda _name: Algorithm)
    result = utils.get_aux_model("checkpoint.ckpt", registry, object(), with_grads=True)
    assert all(parameter.requires_grad for parameter in result.parameters())
