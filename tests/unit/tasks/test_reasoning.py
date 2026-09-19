import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import main  # noqa: F401 -- registers Hydra resolvers
from algos import algo_registry
from tasks.maze import maze_metrics, shortest_path
from tasks.zebra import clue_holds
from tasks.reasoning import encode_record, get_reasoning_tokenizer, score_record
from tasks.reasoning_data import ArrayUnpickler, is_development, source_records, prepare

ROOT = Path(__file__).resolve().parents[3]
MATRIX = OmegaConf.load(ROOT / "configs/sweeps/sudoku_rank.yaml")


def fixture(task):
    if task.startswith("sudoku"):
        answer = [[((r * 3 + r // 3 + c) % 9) + 1 for c in range(9)] for r in range(9)]
        puzzle = np.array(answer)
        puzzle[:, 1:] = 0
        puzzle = puzzle.tolist()
        p, length = 91, 180
    elif task == "maze_unique":
        puzzle = [[1, 1, 1, 1, 1], [1, 3, 2, 4, 1], [1, 1, 1, 1, 1]]
        answer = [[1, 1, 1, 1, 1], [1, 3, 5, 4, 1], [1, 1, 1, 1, 1]]
        p, length = 17, 32
    else:
        puzzle = ["=", "LHS", "c", "0", "0", "RHS", "n", "0", "0", "CLUE_END"]
        answer = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
        p, length = 32, 44
    return (
        dict(
            task=task,
            puzzle=puzzle,
            answer=answer,
            source_id="fixture",
            group_id="base",
        ),
        p,
        length,
    )


@pytest.mark.parametrize(
    "task", ["sudoku_extreme", "sudoku_shah", "maze_unique", "zebra"]
)
def test_encoding_roundtrip_and_masks(task):
    record, p, length = fixture(task)
    tok = get_reasoning_tokenizer(task)
    encoded = encode_record(record, tok, p, length)
    ids = np.array(encoded["input_ids"])
    loss = np.array(encoded["attention_mask"], bool)
    if task == "zebra":
        first_target = np.flatnonzero(loss)[0]
        assert first_target < p
        assert ids[first_target - 1] == tok.ids["<ANSWER>"]
        assert loss.sum() == np.asarray(record["answer"]).size
        assert not loss[first_target + loss.sum() :].any()
    else:
        assert not loss[:p].any()
    assert np.array_equal(~loss, np.array(encoded["prompt_mask"], bool))
    assert not np.any(loss & np.array(encoded["padding_mask"], bool))
    scores = score_record(ids[loss].tolist(), record, tok)
    assert scores["exact"]
    bad = score_record(np.zeros(loss.sum(), dtype=int).tolist(), record, tok)
    assert not bad["exact"]
    assert not bad.get(
        "valid_solution", bad.get("constraints_satisfied", bad.get("valid_path", False))
    )


def test_zebra_answer_trace_never_enters_prompt():
    record, p, length = fixture("zebra")
    tok = get_reasoning_tokenizer("zebra")
    clean = encode_record(record, tok, p, length)
    record["puzzle"] += ["ANSWER", "secret", "solver", "trace"]
    assert encode_record(record, tok, p, length)["input_ids"] == clean["input_ids"]
    with pytest.raises(ValueError, match="never truncate"):
        encode_record(record, tok, 4, length)


def test_zebra_packs_answer_before_trailing_padding():
    record, p, length = fixture("zebra")
    tok = get_reasoning_tokenizer("zebra")
    encoded = encode_record(record, tok, p, length)
    ids = np.asarray(encoded["input_ids"])
    loss = np.asarray(encoded["attention_mask"], dtype=bool)
    target_positions = np.flatnonzero(loss)
    assert np.array_equal(
        target_positions,
        np.arange(target_positions[0], target_positions[0] + len(target_positions)),
    )
    assert ids[target_positions[0] - 1] == tok.ids["<ANSWER>"]
    assert np.all(ids[target_positions[-1] + 1 :] == tok.pad_token_id)


@pytest.mark.parametrize(
    "relation,positions,expected",
    [
        ("=", [1, 1], True),
        ("!=", [1, 1], False),
        ("immediate-left", [1, 2], True),
        ("immediate-left", [2, 1], False),
        ("nbr", [2, 1], True),
        ("ends", [2], True),
        ("ends", [1], False),
        ("left-of", [0, 2], True),
        ("inbetween", [0, 1, 2], True),
        ("inbetween", [2, 1, 0], False),
        ("inbetween", [1, 0, 2], False),
    ],
)
def test_zebra_relations(relation, positions, expected):
    assert (
        clue_holds((relation, [("n", 0, v) for v in positions]), [[0, 1, 2]])
        == expected
    )


def test_maze_rejects_disconnected_or_illegal_cells():
    board = np.array([[3, 2, 4], [1, 1, 1], [2, 2, 2]])
    gold = board.copy()
    gold[0, 1] = 5
    assert maze_metrics(gold, gold, board)["optimal_path"]
    bad = gold.copy()
    bad[2, 0] = 5
    assert not maze_metrics(bad, gold, board)["valid_path"]
    bad = gold.copy()
    bad[1, 0] = 5
    assert not maze_metrics(bad, gold, board)["valid_path"]
    assert not shortest_path([[3, 2], [2, 4]])[1]


def test_restricted_unpickler():
    assert np.array_equal(
        ArrayUnpickler(io.BytesIO(pickle.dumps(np.arange(3)))).load(), np.arange(3)
    )
    with pytest.raises(pickle.UnpicklingError):
        ArrayUnpickler(io.BytesIO(pickle.dumps(Path("/tmp")))).load()


def test_group_assignment_deterministic():
    assert is_development("same", 42, 0.2) == is_development("same", 42, 0.2)
    assert any(is_development(str(i), 42, 0.2) for i in range(30))
    assert not all(is_development(str(i), 42, 0.2) for i in range(30))


def test_zebra_preparation_removes_test_groups_from_training(tmp_path):
    raw = tmp_path / "raw/shah"
    raw.mkdir(parents=True)
    table = np.asarray(
        [[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int64
    )

    def record(house):
        sequence = [
            "=", "LHS", "c", "0", "0", "RHS", "n", "0", str(house),
            "CLUE_END", "ANSWER",
        ]
        return [sequence, table, [[0, 0], [0, 1], [0, 2]]]

    with (raw / "zebra-train-data.pkl").open("wb") as stream:
        pickle.dump([record(0), record(1)], stream)
    with (raw / "zebra-test-data.pkl").open("wb") as stream:
        pickle.dump([record(0)], stream)

    manifest = prepare(
        tmp_path / "raw",
        tmp_path / "prepared",
        "zebra",
        layout=[3, 3],
        full_training=True,
        audit_labels=False,
    )
    assert manifest["counts"] == {"train": 1, "validation": 0, "test": 1}
    assert manifest["decontamination"] == {
        "train_against_test": True,
        "excluded_train_records": 1,
        "excluded_train_groups": 1,
    }
    assert manifest["exact_puzzle_overlap"]["train:test"] == 0


@pytest.mark.parametrize(
    "task", ["sudoku_extreme", "sudoku_shah", "maze_unique", "zebra"]
)
@pytest.mark.parametrize(
    "variant,mode", [(v, m) for v, e in MATRIX.variants.items() for m in e.modes]
)
def test_all_variants_task_backward_and_conditioning(task, variant, mode):
    record, p, length = fixture(task)
    tok = get_reasoning_tokenizer(task)
    encoded = encode_record(record, tok, p, length)
    entry = MATRIX.variants[variant]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                f"+experiment={entry.experiment}",
                *entry.overrides,
                f"data={task}",
                f"algo.pure_continuous={str(mode == 'pure').lower()}",
                f"model.length={length}",
                "model.hidden_size=16",
                "model.cond_dim=16",
                "model.n_blocks=1",
                "model.n_heads=1",
                "model.dropout=0",
                "training.ema=0",
                "wandb=null",
            ],
        )
    model = algo_registry.get(config.algo.name)(config, tok)
    ids = torch.tensor([encoded["input_ids"]] * 2)
    mask = torch.tensor([encoded["attention_mask"]] * 2)
    loss = model._loss(ids, mask).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    model.eval()
    result = model.generate_samples(
        2, num_steps=2, prompt_tokens=ids, prompt_mask=~mask.bool(), return_result=True
    )
    assert torch.equal(result.samples[~mask.bool()], ids[~mask.bool()])
    assert result.samples.max() < len(tok)


def test_preparation_groups_and_manifest_roundtrip(tmp_path):
    record, _, _ = fixture("sudoku_extreme")
    raw = tmp_path / "raw"
    source = raw / "eqr/sudoku-extreme-1k-aug-1000"
    answer = np.asarray(record["answer"])
    for split in ("train", "test"):
        directory = source / split
        directory.mkdir(parents=True)
        n = 90 if split == "train" else 1
        inputs = []
        for i in range(n):
            puzzle = np.zeros((9, 9), dtype=int)
            cell = i // 3 if split == "train" else 80
            puzzle.flat[cell] = answer.flat[cell]
            inputs.append(puzzle.reshape(-1) + 1)
        np.save(directory / "all__inputs.npy", inputs)
        np.save(
            directory / "all__labels.npy", np.tile(answer.reshape(1, -1) + 1, (n, 1))
        )
        np.save(directory / "all__puzzle_indices.npy", np.arange(n + 1))
        np.save(
            directory / "all__group_indices.npy",
            np.arange(0, n + 1, 3) if split == "train" else [0, 1],
        )
    output = tmp_path / "prepared"
    manifest = prepare(raw, output, "sudoku_extreme", dev_fraction=0.2)
    assert manifest["counts"]["train"] + manifest["counts"]["validation"] == 90
    assert manifest["counts"]["validation"] % 3 == 0
    assert manifest["exact_puzzle_overlap"]["train:validation"] == 0
    assert manifest["gold_checks"] == manifest["counts"]
    ids = [
        json.loads(line)
        for line in (output / "selected_ids.jsonl").read_text().splitlines()
    ]
    train = {r["group_id"] for r in ids if r["split"] == "train"}
    validation = {r["group_id"] for r in ids if r["split"] == "validation"}
    assert not train & validation
    with pytest.raises(FileExistsError):
        prepare(raw, output, "sudoku_extreme")
    refit = prepare(raw, tmp_path / "refit", "sudoku_extreme", full_training=True)
    assert refit["counts"]["train"] == 90 and refit["counts"]["validation"] == 0


def test_shah_reconstruction_does_not_leak_solutions(tmp_path):
    (tmp_path / "shah").mkdir()
    answer = np.asarray(fixture("sudoku_shah")[0]["answer"])
    cells = [
        [r, c, int(answer[r, c]), 0 if r == 0 else 1]
        for r in range(9)
        for c in range(9)
    ]
    np.save(
        tmp_path / "shah/Sudoku-train-data.npy",
        np.array([[9] + np.array(cells).reshape(-1).tolist()]),
    )
    record = next(source_records(tmp_path, "sudoku_shah", "train"))
    assert np.count_nonzero(record["puzzle"]) == 9
    assert np.array_equal(record["answer"], answer)
    assert record["puzzle"][1:] == [[0] * 9 for _ in range(8)]


def test_zebra_solver_unique_ambiguous_and_inconsistent():
    from tasks.zebra import solve_clues

    fixed = [("=", [("c", 0, v), ("n", 0, v)]) for v in range(3)]
    assert solve_clues(fixed, 1, 3) == [[[0, 1, 2]]]
    assert len(solve_clues([], 1, 3)) == 2
    assert solve_clues(fixed + [("!=", [("c", 0, 0), ("n", 0, 0)])], 1, 3) == []


def test_reasoning_sweep_matrix_and_layouts(monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(ROOT / "new_scripts"))
    runner = importlib.import_module("reasoning_ablation")
    profiles = OmegaConf.load(ROOT / "configs/sweeps/reasoning_tasks.yaml").profiles
    args = SimpleNamespace(
        profiles=None, variants=None, seeds=None, modes=None, command="plan"
    )
    jobs = list(runner.jobs(MATRIX, profiles, args))
    assert len(jobs) == 342
    assert len({(j["variant"], j["mode"]) for j in jobs}) == 19
    for name, profile in profiles.items():
        job = dict(profile=name, variant="candi_geometric", mode="hybrid", seed=1)
        _, values = runner.overrides(MATRIX, profiles, job, tmp_path, tmp_path / "data")
        assert not any(v.startswith("data.difficulty=") for v in values)
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            config = compose(config_name="config", overrides=values)
        assert config.data.train == profile.task
        assert config.model.length == profile.length
    assert list(profiles.sudoku_shah.evaluations) == ["sudoku_shah", "sudoku_extreme"]
    assert profiles.sudoku_extreme.full_training and profiles.maze_unique.full_training


def test_full_sweep_rejects_smoke_subset(monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(ROOT / "new_scripts"))
    runner = importlib.import_module("reasoning_ablation")
    profiles = OmegaConf.load(ROOT / "configs/sweeps/reasoning_tasks.yaml").profiles
    directory = tmp_path / "prepared/sudoku_extreme"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(dict(source_limit=256)))
    args = SimpleNamespace(
        command="train", prepared=tmp_path / "prepared", root=tmp_path / "runs"
    )
    with pytest.raises(ValueError, match="smoke subset"):
        runner.run(
            MATRIX,
            profiles,
            dict(
                profile="sudoku_extreme",
                variant="candi_geometric",
                mode="hybrid",
                seed=1,
            ),
            args,
            "0",
        )
