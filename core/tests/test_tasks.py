"""Sudoku Task preserves supervision and never leaks answers into generation."""

from dataclasses import asdict

from hydra.utils import instantiate
import torch

from core.tests.test_mdlm import config
from core.tasks.sudoku import SudokuTask
from core.tasks.sudoku_metrics import SudokuRecord, SudokuAggregate


def task_and_examples(tmp_path):
    settings = config()
    settings.data.cache_dir = str(tmp_path)
    task = instantiate(settings.task)
    task.prepare_data()
    return task, [task.validation_dataset()[i] for i in range(2)]


def test_sudoku_training_masks_and_generation_payload(tmp_path):
    task, examples = task_and_examples(tmp_path)
    batch = task.collate_training(examples)
    expected = torch.stack([e["attention_mask"] for e in examples]).bool()
    assert torch.equal(batch["loss_mask"], expected)
    assert torch.equal(batch["corruption_mask"], expected)
    assert torch.equal(batch["prompt_mask"], ~expected)
    assert not batch["padding_mask"].any()  # Blank puzzle cells are not padding.
    prompt, fixed = task.prepare_generation(examples)
    assert torch.equal(fixed, ~expected)
    assert prompt[~fixed].eq(0).all()
    assert torch.equal(prompt[fixed], batch["input_ids"][fixed])
    changed = [{**e, "input_ids": e["input_ids"].clone()} for e in examples]
    for example in changed:
        example["input_ids"][91:] = 7
    other_prompt, other_fixed = task.prepare_generation(changed)
    assert torch.equal(prompt, other_prompt)
    assert torch.equal(fixed, other_fixed)


def test_sudoku_metrics_aggregate_examples_not_batch_means(tmp_path):
    task, examples = task_and_examples(tmp_path)
    metric = task.create_metric_accumulator("validation")
    correct = examples[0]["input_ids"]
    wrong = correct.clone()
    wrong[91:] = 0
    metric.update(torch.stack([correct, correct]), [examples[0]] * 2, [0, 1])
    metric.update(wrong[None], [examples[0]], [2])
    result = metric.compute()
    assert isinstance(result, SudokuAggregate)
    assert all(isinstance(record, SudokuRecord) for record in result.records)
    assert asdict(result)["records"][0]["exact"] is True
    assert result.accuracy == 2 / 3
    assert result.num_correct == 2
    assert result.num_total == 3
    assert task.primary_metric.name == "accuracy"
    assert task.primary_metric.mode == "max"


def test_one_task_instance_is_shared_with_infrastructure(tmp_path):
    task, _ = task_and_examples(tmp_path)
    settings = config()
    data = instantiate(settings.datamodule, task=task)
    callback = instantiate(settings.callback, task=task)
    assert data.task is task
    assert callback.task is task
    assert isinstance(task, SudokuTask)


def test_task_collation_preserves_training_loss_and_gradients(tmp_path):
    task, examples = task_and_examples(tmp_path)
    algorithm = instantiate(config().algorithm)
    batch = task.collate_training(examples)
    legacy_mask = torch.stack([e["attention_mask"] for e in examples])
    torch.manual_seed(19)
    reference = algorithm.loss(batch["input_ids"], legacy_mask)["loss"]
    reference.backward()
    gradients = {
        name: p.grad.clone()
        for name, p in algorithm.named_parameters()
        if p.grad is not None
    }
    algorithm.zero_grad(set_to_none=True)
    torch.manual_seed(19)
    actual = algorithm.loss(
        batch["input_ids"], batch["loss_mask"], batch["corruption_mask"]
    )["loss"]
    actual.backward()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for name, p in algorithm.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
