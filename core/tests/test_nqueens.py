"""Rule validity, leakage prevention and reproducible N-Queens integration."""

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.data.nqueens import (
    enumerate_solutions,
    get_nqueens_dataset,
    is_valid_solution,
)
from core.tasks.nqueens import NQueensTask
from core.tasks.nqueens_metrics import NQueensMetrics
from core.tests.test_mdlm import config


def settings(tmp_path, size=8):
    return OmegaConf.create(
        dict(
            size=size,
            cache_dir=str(tmp_path),
            num_train=12,
            num_valid=8,
            data_seed=42,
            min_clues=2,
            max_clues=4,
            validation_solution_fraction=0.2,
        )
    )


def test_exhaustive_solutions_and_invalid_boards():
    for size, expected in [(8, 92), (10, 724)]:
        solutions = enumerate_solutions(size)
        assert len(solutions) == expected
        assert all(is_valid_solution(x, size) for x in solutions)
    assert not is_valid_solution(list(range(1, 9)), 8)
    assert not is_valid_solution([1] * 8, 8)
    assert not is_valid_solution([0] * 8, 8)
    assert not is_valid_solution(enumerate_solutions(8)[0] + (1,), 8)


def test_cache_disjoint_prompts_and_reference_solutions(tmp_path):
    data = get_nqueens_dataset(settings(tmp_path))
    again = get_nqueens_dataset(settings(tmp_path))
    rows = {}
    for split in data:
        rows[split] = [tuple(x["input_ids"].tolist()) for x in data[split]]
        assert rows[split] == [tuple(x["input_ids"].tolist()) for x in again[split]]
    assert not {x[:8] for x in rows["train"]} & {x[:8] for x in rows["validation"]}
    assert not {x[8:] for x in rows["train"]} & {x[8:] for x in rows["validation"]}
    uncached = settings(tmp_path / "other")
    new = get_nqueens_dataset(uncached)
    assert rows["train"] == [tuple(x["input_ids"].tolist()) for x in new["train"]]


def test_masks_and_unconditional_generation_have_no_answer_leakage(tmp_path):
    task = NQueensTask(settings(tmp_path), 9, 16, "unconditional", 3)
    examples = task.evaluation_dataset("validation")
    assert len(examples) == 24
    prompt, fixed = task.prepare_generation(examples[:2])
    assert not prompt.any()
    assert fixed[:, :8].all() and not fixed[:, 8:].any()
    batch = task.collate_training([task.training_dataset()[0]])
    assert batch["loss_mask"][:, 8:].all()
    assert not batch["loss_mask"][:, :8].any()
    assert not batch["padding_mask"].any()
    changed = [{"input_ids": x["input_ids"].clone()} for x in examples[:2]]
    for x in changed:
        x["input_ids"][8:] = 5
    assert torch.equal(task.prepare_generation(changed)[0], prompt)


def test_alternative_valid_answers_and_clues_and_diversity():
    first, second = enumerate_solutions(8)[:2]
    example = {"input_ids": torch.tensor([0] * 8 + list(first))}
    metric = NQueensMetrics(8, 92)
    samples = torch.tensor([[0] * 8 + list(x) for x in [first, second, second]])
    metric.update(samples, [example] * 3, [0, 0, 0])
    result = metric.compute()
    assert result.valid_solution_rate == 1
    assert result.exact_match_rate == 1 / 3
    assert result.unique_valid_solutions == 2
    assert result.unique_valid_fraction == 2 / 3
    assert result.mean_unique_valid_per_prompt == 2
    assert result.solution_coverage == 2 / 92
    row = next(i for i in range(8) if first[i] != second[i])
    example["input_ids"][row] = first[row]
    metric = NQueensMetrics(8, 92)
    metric.update(samples[1:2], [example], [0])
    assert metric.compute().valid_board_rate == 1
    assert metric.compute().valid_solution_rate == 0


def test_nqueens_task_runs_loss_and_generation(tmp_path):
    cfg = config(
        "task=nqueens",
        f"data.cache_dir={tmp_path}",
        "sampler=ancestral",
        "sampler.steps=2",
    )
    task = instantiate(cfg.task)
    algorithm = instantiate(cfg.algorithm)
    examples = [task.validation_dataset()[0]]
    batch = task.collate_training(examples)
    loss = algorithm.loss(
        batch["input_ids"], batch["loss_mask"], batch["corruption_mask"]
    )["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    prompt, fixed = task.prepare_generation(examples)
    samples, metadata = algorithm.generate(instantiate(cfg.sampler), prompt, fixed)
    metric = task.create_metric_accumulator("validation")
    metric.update(samples, examples, [0])
    assert metric.compute().num_total == 1
    assert metadata["nfe_per_sample"] == [3]
