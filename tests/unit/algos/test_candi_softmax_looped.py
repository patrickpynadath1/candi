from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- config resolvers
from algos.candi_softmax_looped import CANDISoftmaxLooped
from dataloader import Text8Tokenizer
from models.dit_cont import ContDIT


def _config(*overrides):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        return compose(config_name="config", overrides=[
            "algo=candi_softmax_looped", "model=tiny", "data=ptb",
            "sampling.predictor=candi_looped_entropy_bounded",
            "training.ema=0", "model.length=4", "model.hidden_size=8",
            "model.cond_dim=8", "model.n_blocks=1", "model.n_heads=2",
            "model.dropout=0", *overrides,
        ])


@pytest.mark.parametrize("input_dim", [6, 8])
def test_feedback_is_softmax_weighted_input_table_added_before_projection(input_dim):
    model = CANDISoftmaxLooped(
        _config(f"+algo.input_embedding_dim={input_dim}"), Text8Tokenizer()
    )
    backbone = model.backbone
    embeddings = torch.randn(2, 4, input_dim)
    probabilities = torch.randn(2, 4, model.vocab_size - 1).softmax(-1)
    actual = backbone._condition_hidden(
        backbone.input_projection(embeddings), probabilities
    )
    expected = backbone.input_projection(
        embeddings + probabilities @ backbone.vocab_embed.embedding[:-1]
    )
    assert torch.allclose(actual, expected, atol=1e-6)
    with pytest.raises(ValueError, match="data_vocab_size"):
        backbone._condition_hidden(
            backbone.input_projection(embeddings), probabilities[:, :, :-1]
        )


def test_first_pass_matches_base_and_returns_raw_softmax_as_carry():
    config = _config()
    model = CANDISoftmaxLooped(config, Text8Tokenizer()).eval()
    base = ContDIT(config, model.vocab_size).eval()
    base.load_state_dict(model.backbone.state_dict(), strict=True)
    noisy = model.q_xt(torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 1))
    with torch.no_grad():
        logits, carry = model.backbone(**noisy, ret_embedding=True)
        assert torch.equal(logits, base(**noisy))
        assert torch.equal(carry, logits.float().softmax(-1))
        assert torch.allclose(carry.sum(-1), torch.ones(1, 4))
        assert not torch.allclose(
            logits, model.backbone(**noisy, previous_hidden=carry)
        )


def test_training_detaches_predictions_but_feedback_table_receives_gradient():
    model = CANDISoftmaxLooped(_config(), Text8Tokenizer())
    calls = []

    def record(_module, args, kwargs, output):
        output[0].retain_grad()
        calls.append((kwargs, output))

    handle = model.backbone.register_forward_hook(record, with_kwargs=True)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 5], [5, 4, 3, 2]])
    noisy = model.q_xt(tokens, torch.full((4, 1), 0.2))
    noisy["reveal_mask"].zero_()
    noisy["example_ids"] = torch.arange(4)
    predictions = model._denoise_for_loss(noisy, train_mode=True)
    (-predictions.gather(-1, tokens[..., None]).mean()).backward()
    handle.remove()
    assert len(calls) == 2
    first_inputs, (first_logits, first_carry) = calls[0]
    second_inputs, (second_logits, _) = calls[1]
    selected = second_inputs["example_ids"]
    assert len(selected.unique()) == 2
    for key in noisy:
        assert torch.equal(second_inputs[key], first_inputs[key][selected])
    assert torch.equal(second_inputs["previous_hidden"], first_carry[selected])
    assert not second_inputs["previous_hidden"].requires_grad
    assert torch.count_nonzero(first_logits.grad[selected]) == 0
    assert torch.count_nonzero(first_logits.grad) > 0
    assert torch.count_nonzero(second_logits.grad) > 0

    # Isolate the feedback branch from ordinary input-embedding gradients.
    model.zero_grad(set_to_none=True)
    model.backbone._condition_hidden(
        torch.zeros(2, 4, 8), second_inputs["previous_hidden"]
    ).square().sum().backward()
    grad = model.backbone.vocab_embed.embedding.grad
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[:-1]) > 0
    assert torch.count_nonzero(grad[-1]) == 0


def test_sampling_carries_previous_probabilities_one_pass_per_step():
    model = CANDISoftmaxLooped(_config(), Text8Tokenizer()).eval()
    calls = []
    handle = model.backbone.register_forward_hook(
        lambda _module, args, kwargs, output: calls.append((kwargs, output[1])),
        with_kwargs=True,
    )
    for _ in range(2):
        result = model.generate_samples(
            1, num_steps=3, sampler_name="candi_looped", return_result=True
        )
        assert result.metadata["nfe"] == 3
    handle.remove()
    assert len(calls) == 6
    for index, (inputs, carry) in enumerate(calls):
        assert carry.shape == (1, 4, model.vocab_size - 1)
        if index % 3 == 0:
            assert inputs["previous_hidden"] is None
        else:
            assert torch.equal(inputs["previous_hidden"], calls[index - 1][1])


def test_sudoku_preset_selects_hard_and_eb_greedy():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        config = compose(config_name="config", overrides=[
            "+experiment=sudoku_candi_softmax_looped",
        ])
    assert config.data.difficulty == "hard"
    assert config.sampling.predictor == "candi_looped_entropy_bounded"
    assert config.sampling.greedy_clean_estimate
    assert config.sampling.entropy_bound == 0.1
