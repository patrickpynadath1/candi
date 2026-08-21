from types import SimpleNamespace

import torch

from samplers.base import SamplingResult
from tasks.sudoku import SudokuTokenizer, encode_grid
from tasks.sudoku_callback import (
    SudokuEvaluationCallback,
    generate_sudoku_records,
    summarize_sudoku_records,
)


def _solution_tokens():
    grid = [
        [((row * 3 + row // 3 + column) % 9) + 1 for column in range(9)]
        for row in range(9)
    ]
    return torch.tensor(encode_grid(grid))


class PerfectSudokuModel:
    device = torch.device("cpu")
    num_tokens = SudokuTokenizer.total_seq_len

    def __init__(self, solution):
        self.solution = solution
        self.eval_calls = 0
        self.train_calls = 0

    def generate_samples(self, count, **kwargs):
        result = kwargs["prompt_tokens"].clone()
        start = SudokuTokenizer.prompt_len
        result[:, start : start + len(self.solution)] = self.solution
        if kwargs.get("return_result"):
            return SamplingResult(
                result,
                {"nfe": kwargs.get("num_steps", 1)},
            )
        return result

    def _eval_mode(self):
        self.eval_calls += 1

    def _train_mode(self):
        self.train_calls += 1


def _dataset(count=2):
    solution = _solution_tokens()
    sequence = torch.zeros(SudokuTokenizer.total_seq_len, dtype=torch.long)
    sequence[SudokuTokenizer.prompt_len :] = solution
    return [{"input_ids": sequence.clone()} for _ in range(count)]


def test_generation_and_summary_report_solve_rate():
    tokenizer = SudokuTokenizer()
    solution = _solution_tokens()
    records = generate_sudoku_records(
        PerfectSudokuModel(solution),
        _dataset(3),
        tokenizer,
        batch_size=2,
        num_steps=4,
        indices=[0, 2],
    )
    results = summarize_sudoku_records(records)

    assert [record["index"] for record in records] == [0, 2]
    assert results["solve_rate"] == 1.0
    assert results["cell_accuracy"] == 1.0
    assert results["valid_grid_rate"] == 1.0
    assert results["num_correct"] == results["num_total"] == 2
    assert results["mean_nfe"] == 4


def test_callback_runs_fixed_subset_logs_and_restores_training(monkeypatch, tmp_path):
    dataset = _dataset(4)
    monkeypatch.setattr(
        "tasks.sudoku_callback.get_sudoku_dataset",
        lambda _config, _tokenizer: {"validation": dataset},
    )

    class Logger:
        def __init__(self):
            self.metrics = None

        def log_metrics(self, metrics, step):
            self.metrics = (metrics, step)

    class Strategy:
        def __init__(self):
            self.barriers = []

        def barrier(self, name):
            self.barriers.append(name)

    logger = Logger()
    strategy = Strategy()
    trainer = SimpleNamespace(
        global_step=10, is_global_zero=True, logger=logger, strategy=strategy
    )
    model = PerfectSudokuModel(_solution_tokens())
    model.config = SimpleNamespace()
    model.tokenizer = SudokuTokenizer()
    callback = SudokuEvaluationCallback(
        every_n_train_steps=10,
        subset_size=2,
        batch_size=2,
        num_steps=2,
        seed=3,
        output_dir=tmp_path,
    )

    callback.on_train_batch_end(trainer, model, None, None, 0)

    metrics, step = logger.metrics
    assert step == 10
    assert metrics["sudoku/subset/solve_rate"] == 1.0
    assert metrics["sudoku/subset/num_total"] == 2
    assert model.eval_calls == model.train_calls == 1
    assert strategy.barriers == ["sudoku-subset-evaluation"]
    assert len(list(tmp_path.glob("step_00000010_subset.json"))) == 1

    # Repeated hooks at the same global step must not evaluate twice.
    callback.on_train_batch_end(trainer, model, None, None, 1)
    assert model.eval_calls == 1


def test_callback_evaluates_multiple_named_samplers(monkeypatch, tmp_path):
    dataset = _dataset(2)
    monkeypatch.setattr(
        "tasks.sudoku_callback.get_sudoku_dataset",
        lambda _config, _tokenizer: {"validation": dataset},
    )

    class Logger:
        def __init__(self):
            self.metrics = []

        def log_metrics(self, metrics, step):
            self.metrics.append((metrics, step))

    trainer = SimpleNamespace(
        global_step=5,
        is_global_zero=True,
        logger=Logger(),
        strategy=SimpleNamespace(barrier=lambda _name: None),
    )
    model = PerfectSudokuModel(_solution_tokens())
    model.config = SimpleNamespace()
    model.tokenizer = SudokuTokenizer()
    callback = SudokuEvaluationCallback(
        every_n_train_steps=5,
        subset_size=2,
        batch_size=2,
        num_steps=10,
        output_dir=tmp_path,
        samplers=[
            {"name": "ancestral", "predictor": "ancestral_cache", "num_steps": 4},
            {
                "name": "eb_01",
                "predictor": "uniform_entropy_bounded",
                "num_steps": 7,
                "entropy_bound": 0.1,
            },
        ],
    )

    callback.on_train_batch_end(trainer, model, None, None, 0)

    combined = {key: value for metrics, _ in trainer.logger.metrics for key, value in metrics.items()}
    assert combined["sudoku/subset/ancestral/solve_rate"] == 1.0
    assert combined["sudoku/subset/ancestral/mean_nfe"] == 4
    assert combined["sudoku/subset/eb_01/mean_nfe"] == 7
    assert (tmp_path / "step_00000005_subset_ancestral.json").exists()
    assert (tmp_path / "step_00000005_subset_eb_01.json").exists()
    assert model.eval_calls == model.train_calls == 1
