from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- register Hydra resolvers
from algos import algo_registry
from algos.frm import GRID_LENGTH, GRID_VOCAB_SIZE, SOLUTION_START, _grid_indices
from tasks.sudoku import SudokuTokenizer, encode_example


CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def tiny_config(stage="self_condition"):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="config",
            overrides=[
                "algo=frm",
                "model=frm_sudoku",
                "data=sudoku_extreme",
                "strategy=single_cpu",
                "training.ema=0",
                "model.hidden_size=16",
                "model.cond_dim=16",
                "model.n_blocks=1",
                "model.n_heads=2",
                "model.dropout=0",
                f"algo.training_stage={stage}",
                "algo.supervised_states=1",
                "algo.fpf_rollout_depth=3",
                "sampling.predictor=frm_euler",
                "sampling.steps=4",
            ],
        )


def serialized_example():
    solution = [[(row * 3 + row // 3 + column) % 9 + 1 for column in range(9)] for row in range(9)]
    puzzle = [[value if (row + column) % 4 == 0 else 0 for column, value in enumerate(values)] for row, values in enumerate(solution)]
    ids, mask = encode_example(puzzle, solution, SudokuTokenizer())
    return torch.tensor([ids]), torch.tensor([mask]), puzzle, solution


def test_frm_loss_uses_81_cell_network_and_backpropagates():
    model = algo_registry.get("frm")(tiny_config(), SudokuTokenizer())
    tokens, valid, _, _ = serialized_example()
    loss = model._loss(tokens, valid, train_mode=True)
    loss.loss.backward()

    assert loss.num_tokens.item() == GRID_LENGTH
    assert torch.isfinite(loss.loss)
    assert model.backbone.output_layer.linear.weight.grad is not None
    assert model.backbone.output_layer.linear.out_features == GRID_VOCAB_SIZE


def test_fpf_rollout_is_detached_and_preserves_clues():
    model = algo_registry.get("frm")(tiny_config("fpf"), SudokuTokenizer())
    tokens, _, _, _ = serialized_example()
    clues, target = model.unpack_training_batch(tokens)
    noise = torch.randn(1, GRID_LENGTH, GRID_VOCAB_SIZE)
    carry = model.fpf_rollout(
        noise, target, clues, torch.tensor([0.1]), torch.tensor([0.8])
    )

    fixed = clues.ne(0)
    assert not carry.requires_grad
    assert model.backbone.training
    assert torch.equal(carry.argmax(-1)[fixed], clues[fixed])


def test_frm_sampler_returns_transport_layout_and_exact_nfe():
    model = algo_registry.get("frm")(tiny_config(), SudokuTokenizer()).eval()
    tokens, _, puzzle, _ = serialized_example()
    prompt = torch.zeros_like(tokens)
    prompt[:, :91] = tokens[:, :91]
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :91] = True
    result = model.generate_samples(
        1,
        num_steps=4,
        prompt_tokens=prompt,
        prompt_mask=mask,
        return_result=True,
    )

    generated_grid = result.samples.index_select(
        1, _grid_indices(SOLUTION_START)
    ).reshape(9, 9)
    puzzle = torch.tensor(puzzle)
    assert result.samples.shape == (1, 180)
    assert result.metadata["nfe"] == 4
    assert torch.equal(generated_grid[puzzle.ne(0)], puzzle[puzzle.ne(0)])
    assert torch.equal(result.samples[:, :91], prompt[:, :91])


def test_loss_bearing_state_is_not_clamped_to_ground_truth_path(monkeypatch):
    model = algo_registry.get("frm")(tiny_config(), SudokuTokenizer())
    tokens, _, _, _ = serialized_example()
    clues, target = model.unpack_training_batch(tokens)
    seen = {}

    def capture_forward(state, time, observed_clues, carry=None):
        seen["state"] = state.detach().clone()
        return torch.zeros(*state.shape[:-1], GRID_VOCAB_SIZE, requires_grad=True)

    monkeypatch.setattr(model, "forward", capture_forward)
    model.self_condition_probability = 0.0
    model._state_loss(clues, target, train_mode=True)
    fixed = clues.ne(0)
    fixed_state = seen["state"][fixed]
    clue_onehot = torch.nn.functional.one_hot(
        clues[fixed], GRID_VOCAB_SIZE
    ).float()
    assert not torch.equal(fixed_state, clue_onehot)
