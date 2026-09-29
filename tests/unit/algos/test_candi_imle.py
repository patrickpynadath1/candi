from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- config resolvers
from algos.candi_imle import CANDIIMLE, closest_candidate_loss
from dataloader import Text8Tokenizer


def _config(*overrides):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        return compose(config_name="config", overrides=[
            "algo=candi_imle", "sampling.predictor=candi_imle",
            "sampling.greedy_clean_estimate=true",
            "model=tiny", "data=ptb", "training.ema=0",
            "model.length=4", "model.hidden_size=8", "model.cond_dim=8",
            "model.n_blocks=1", "model.n_heads=2", "model.dropout=0",
            *overrides,
        ])


def test_selects_whole_sequence_per_example_and_only_winners_get_gradient():
    losses = torch.tensor([
        [[900., 0., 4.], [0., 1., 3.]],
        [[0., 3., 2.], [999., 2., 0.]],
    ], requires_grad=True)
    valid = torch.tensor([[0, 1, 1], [0, 1, 1]])
    selected, winners = closest_candidate_loss(losses, valid)
    assert winners.tolist() == [0, 1]
    assert torch.equal(selected, torch.stack((losses[0, 0], losses[1, 1])))
    (selected * valid).sum().backward()
    expected_grad = torch.zeros_like(losses)
    expected_grad[0, 0, 1:] = 1
    expected_grad[1, 1, 1:] = 1
    assert torch.equal(losses.grad, expected_grad)


def test_zero_target_positions_and_ties_are_finite():
    losses = torch.ones(3, 2, 4, requires_grad=True)
    selected, winners = closest_candidate_loss(losses, torch.zeros(2, 4))
    assert winners.tolist() == [0, 0]
    assert torch.isfinite(selected).all()


@pytest.mark.parametrize("train_mode,expected_calls", [(True, 8), (False, 1)])
def test_candidate_count_prompt_preservation_independent_noise_and_backward(
    train_mode, expected_calls
):
    model = CANDIIMLE(_config(), Text8Tokenizer())
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    valid = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    calls = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    loss = model._loss(tokens, valid, train_mode=train_mode).loss
    loss.backward()
    handle.remove()
    assert torch.isfinite(loss)
    assert len(calls) == expected_calls
    expected_prompt = torch.nn.functional.one_hot(tokens[:, :2], model.vocab_size - 1)
    for call in calls:
        assert torch.equal(call["xt"][:, :2], expected_prompt)
        assert torch.equal(call["reveal_mask"], ~valid.bool())
        assert torch.equal(call["discrete_noise"], torch.ones(2))
    if train_mode:
        assert not torch.equal(calls[0]["xt"][:, 2:], calls[1]["xt"][:, 2:])
    gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)


def test_candidate_inputs_do_not_depend_on_target_tokens():
    model = CANDIIMLE(_config(), Text8Tokenizer())
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 8, 9]])
    valid = torch.tensor([[0, 0, 1, 1]])
    calls = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append(kwargs["xt"].clone()),
        with_kwargs=True,
    )
    for tokens in (first, second):
        torch.manual_seed(13)
        model.nll(tokens, None, train_mode=True, valid_tokens=valid)
    handle.remove()
    assert len(calls) == 16
    assert all(torch.equal(a, b) for a, b in zip(calls[:8], calls[8:]))


def test_k_one_is_plain_noise_conditioned_cross_entropy():
    model = CANDIIMLE(_config("algo.imle_candidates=1"), Text8Tokenizer())
    tokens = torch.tensor([[1, 2, 3, 4]])
    valid = torch.ones_like(tokens)
    torch.manual_seed(5)
    loss = model.nll(tokens, None, train_mode=True, valid_tokens=valid)
    torch.manual_seed(5)
    state = model.prior_sample(1, 4)
    expected = -model.sample_imle_distribution(
        state, torch.zeros_like(tokens, dtype=torch.bool)
    ).gather(-1, tokens[..., None]).squeeze(-1)
    assert torch.equal(loss, expected)


@pytest.mark.parametrize("greedy", [True, False])
def test_sampling_is_one_target_free_pass_and_preserves_prompt(greedy):
    model = CANDIIMLE(_config(
        f"sampling.greedy_clean_estimate={str(greedy).lower()}"
    ), Text8Tokenizer()).eval()
    prompt = torch.tensor([[1, 2, 0, 0]])
    mask = torch.tensor([[True, True, False, False]])
    calls = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    result = model.generate_samples(
        1, num_steps=180, prompt_tokens=prompt, prompt_mask=mask, return_result=True
    )
    handle.remove()
    assert len(calls) == 1
    assert result.metadata["nfe"] == 1
    assert result.metadata["nfe_per_sample"] == [1]
    assert torch.equal(result.samples[mask], prompt[mask])
    assert result.samples.shape == prompt.shape
    assert result.samples.min() >= 0 and result.samples.max() < model.vocab_size - 1


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_rejects_invalid_k(value):
    config = _config()
    config.algo.imle_candidates = value
    with pytest.raises(ValueError, match="imle_candidates"):
        CANDIIMLE(config, Text8Tokenizer())


def test_rejects_adaptive_noise_schedule():
    with pytest.raises(ValueError, match="fixed pure noise"):
        CANDIIMLE(_config("noise=log-linear-adaptive"), Text8Tokenizer())


def test_hard_sudoku_preset_has_eight_candidates_and_single_pass_evaluation():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        config = compose(config_name="config", overrides=["+experiment=sudoku_candi_imle"])
    assert config.data.difficulty == "hard"
    assert config.algo.imle_candidates == 8
    assert config.sampling.predictor == "candi_imle"
    assert config.callbacks.sudoku_eval.num_steps == 1
