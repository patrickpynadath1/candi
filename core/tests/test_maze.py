from dataclasses import asdict
import json
import importlib.util
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from core.data.maze import (
    cell_connections,
    generate_splits,
    load_maze_dataset,
    map_identity,
    neighbors,
)
from core.sweeps import compose
from core.tasks.maze import MazeTask
from core.training.runner import run


def data_config(tmp_path):
    return dict(
        cache_dir=str(tmp_path),
        grid_size=4,
        num_waypoints=1,
        families=["tree", "cyclic"],
        cycle_probability=0.2,
        min_path_edges=2,
        max_path_edges=24,
        num_train=8,
        num_valid=4,
        num_test=4,
        data_seed=17,
    )


def test_generation_cache_disjoint_and_rng(tmp_path, monkeypatch):
    if importlib.util.find_spec("maze_dataset") is None:
        pytest.skip("optional maze-dataset dependency not installed")
    __import__("maze_dataset")
    config = data_config(tmp_path)
    random.seed(5)
    np.random.seed(5)
    expected_py = random.random()
    expected_np = np.random.rand()
    random.seed(5)
    np.random.seed(5)
    splits = generate_splits(config)
    assert random.random() == expected_py and np.random.rand() == expected_np
    assert splits == generate_splits(config)
    identities = [x["maze_id"] for rows in splits.values() for x in rows]
    assert len(identities) == len(set(identities))
    for rows in splits.values():
        for example in rows:
            size = example["grid_size"]
            edges = sum(x.bit_count() for x in example["cells"]) // 2
            assert (
                (edges == size * size - 1)
                if example["family"] == "tree"
                else (edges >= size * size)
            )
            assert all(
                v in neighbors(example["cells"], u, size)
                for u, v in zip(example["path"], example["path"][1:])
            )
    cached = load_maze_dataset(config)
    import core.data.maze as module

    monkeypatch.setattr(
        module, "generate_splits", lambda _: pytest.fail("cache regenerated")
    )
    assert cached == load_maze_dataset(config)


def fixture_task(tmp_path, variable_length=False):
    config = data_config(tmp_path)
    config.update(grid_size=2, num_waypoints=0, max_path_edges=6)
    task = MazeTask(config, vocab_size=36, length=20, variable_length=variable_length)
    example = dict(
        grid_size=2,
        maze_id="square",
        family="cyclic",
        cells=[6, 12, 3, 9],
        anchors=[0, 3],
        path=[0, 1, 3],
    )
    return task, example


def test_masks_and_no_answer_or_length_leak(tmp_path):
    task, example = fixture_task(tmp_path)
    longer = {**example, "path": [0, 1, 0, 2, 3]}
    a, am = task.prepare_generation([example])
    b, bm = task.prepare_generation([longer])
    assert torch.equal(a, b) and torch.equal(am, bm)
    batch = task.collate_training([example])
    assert batch["loss_mask"][0, -1]  # padded slots remain unknown for fixed canvases
    task.variable_length = True
    batch = task.collate_training([example])
    assert not (batch["loss_mask"] & ~batch["attention_mask"]).any()
    assert not batch["loss_mask"][0, -1]


def test_rule_scoring_alternatives_walls_waypoints_and_lengths(tmp_path):
    task, example = fixture_task(tmp_path)
    prefix = task.prompt(example)

    def sample(path, eos=True):
        ids = prefix + [32 + v for v in path] + ([1] if eos else [])
        return torch.tensor(ids + [0] * (task.length - len(ids)))

    metrics = task.create_metric_accumulator("validation")
    metrics.update(
        torch.stack(
            [
                sample([0, 2, 3]),
                sample([0, 1, 3]),
                sample([0, 3]),
                sample([0, 2, 3], False),
            ]
        ),
        [example] * 4,
        range(4),
    )
    result = metrics.compute()
    assert result.success_rate == result.optimal_solution_rate == 0.5
    assert result.mean_unique_valid_routes_per_maze == 2
    assert result.mean_excess_edges_successful == 0
    # Variable-length padded batches must not let EOS outside reported length rescue a sample.
    metrics = task.create_metric_accumulator("validation")
    metrics.update(sample([0, 2, 3])[None], [example], [0], lengths=[len(prefix) + 3])
    assert metrics.compute().success_rate == 0
    # Ordered waypoint must actually be visited, even if endpoints and moves are valid.
    constrained = {**example, "anchors": [0, 1, 3]}
    prefix = task.prompt(constrained)
    metrics = task.create_metric_accumulator("validation")
    metrics.update(sample([0, 2, 3])[None], [constrained], [0])
    assert metrics.compute().valid_route_rate == 1
    assert metrics.compute().success_rate == 0
    assert json.loads(json.dumps(asdict(result)))["num_total"] == 4


def test_map_identity_excludes_symmetries():
    connections = np.array([[[1, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=bool)
    cells = cell_connections(connections)
    reflected = [
        6,
        8,
        3,
        8,
    ]  # a different spanning tree of the square, same up-to-symmetry topology
    assert map_identity(cells, 2) == map_identity(reflected, 2)


@pytest.mark.parametrize(
    "preset", ["maze", "maze_insertion", "maze_waypoints", "maze_waypoints_insertion"]
)
def test_cpu_training_and_evaluation(tmp_path, preset):
    if importlib.util.find_spec("maze_dataset") is None:
        pytest.skip("optional maze-dataset dependency not installed")
    __import__("maze_dataset")
    cfg = compose(preset)
    cfg.data = OmegaConf.create(data_config(tmp_path / "cache"))
    cfg.data.num_waypoints = 1 if "waypoints" in preset else 0
    cfg.task.vocab_size = 48
    cfg.task.length = 64
    cfg.model.backbone.hidden_size = 16
    cfg.model.backbone.cond_dim = 8
    cfg.model.backbone.n_blocks = 1
    cfg.model.backbone.n_heads = 2
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.precision = "32-true"
    cfg.trainer.max_steps = 2
    cfg.trainer.accumulate_grad_batches = 1
    cfg.trainer.val_check_interval = 2
    cfg.trainer.log_every_n_steps = 1
    cfg.loader.global_batch_size = 2
    cfg.loader.eval_batch_size = 2
    cfg.loader.num_workers = 0
    cfg.callback.every_n_steps = 2
    cfg.callback.subset_size = 2
    cfg.sampler.steps = 2
    cfg.output_dir = str(tmp_path / "training")
    run(cfg)
    checkpoint = str(tmp_path / "training/checkpoints/last.ckpt")
    assert (
        json.loads(
            (tmp_path / "training/task_callback/step_00000002.json").read_text()
        )["num_total"]
        == 2
    )
    cfg.mode = "evaluate"
    cfg.evaluation.checkpoint = checkpoint
    cfg.evaluation.device = "cpu"
    cfg.evaluation.max_examples = 2
    OmegaConf.update(cfg, "evaluator.evaluation_name", "test", force_add=True)
    cfg.output_dir = str(tmp_path / "evaluation")
    run(cfg)
    result = json.loads((tmp_path / "evaluation/results.json").read_text())
    assert result["num_total"] == 2
    assert result["checkpoint_step"] == 2
    assert result["primary_metric"]["dataset"] == "test"
    assert "success_rate" in result
