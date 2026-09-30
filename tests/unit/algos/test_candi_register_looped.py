from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- config resolvers
from algos.candi_register_looped import CANDIRegisterLooped
from dataloader import Text8Tokenizer
from samplers import SamplingRequest, create_sampler


def _config(*overrides):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        return compose(config_name="config", overrides=[
            "algo=candi_register_looped", "sampling.predictor=candi_looped",
            "model=tiny", "data=ptb", "training.ema=0",
            "model.length=4", "model.hidden_size=8", "model.cond_dim=8",
            "model.n_blocks=1", "model.n_heads=2", "model.dropout=0",
            *overrides,
        ])


def _open_attention_gates(model):
    # Initial AdaLN gates are zero; exercise register interaction after learning.
    with torch.no_grad():
        for block in model.backbone.blocks:
            block.adaLN_modulation.bias[2 * model.backbone.dim:3 * model.backbone.dim].fill_(1)


@pytest.mark.parametrize("registers", [1, 4, 6])
def test_only_register_states_recur_and_only_data_positions_have_logits(registers):
    model = CANDIRegisterLooped(_config(f"algo.num_registers={registers}"), Text8Tokenizer()).eval()
    _open_attention_gates(model)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    noisy = model.q_xt(tokens, torch.full((2, 1), 0.2))
    with torch.no_grad():
        first, state = model.backbone(**noisy, ret_embedding=True)
        second = model.backbone(**noisy, previous_hidden=state)
    assert first.shape == (2, 4, model.vocab_size - 1)
    assert state.shape == (2, registers, 8)
    assert state.untyped_storage().nbytes() == state.numel() * state.element_size()
    assert not torch.allclose(first, second)
    with pytest.raises(ValueError, match="num_registers"):
        model.backbone(**noisy, previous_hidden=state[:, :-1])
    if registers != tokens.shape[1]:
        with pytest.raises(ValueError, match="num_registers"):
            model.backbone(**noisy, previous_hidden=torch.zeros(2, 4, 8))


def test_training_recomputes_half_with_detached_registers_and_final_pass_loss():
    model = CANDIRegisterLooped(_config("algo.num_registers=2"), Text8Tokenizer())
    _open_attention_gates(model)
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
    first_kwargs, (first_logits, first_state) = calls[0]
    second_kwargs, (second_logits, _) = calls[1]
    selected = second_kwargs["example_ids"]
    assert len(selected) == 2 and len(selected.unique()) == 2
    for key in noisy:
        assert torch.equal(second_kwargs[key], first_kwargs[key][selected])
    assert second_kwargs["previous_hidden"].shape == (2, 2, 8)
    assert torch.equal(second_kwargs["previous_hidden"], first_state[selected])
    assert not second_kwargs["previous_hidden"].requires_grad
    assert torch.count_nonzero(first_logits.grad[selected]) == 0
    assert torch.count_nonzero(first_logits.grad) > 0
    assert torch.count_nonzero(second_logits.grad) > 0
    assert torch.count_nonzero(model.backbone.feedback_projection.weight.grad) > 0
    assert torch.count_nonzero(model.backbone.register_embedding.grad) > 0


def test_sampling_resets_registers_and_preserves_prompt():
    model = CANDIRegisterLooped(_config("algo.num_registers=2"), Text8Tokenizer()).eval()
    calls = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append(kwargs["previous_hidden"]),
        with_kwargs=True,
    )
    sampler = create_sampler(model.config)
    sampler.validate_model(model)
    prompt = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        for _ in range(2):
            result = sampler.sample(model, SamplingRequest(
                1, num_steps=3, prompt_tokens=prompt, prompt_mask=mask
            ))
            assert torch.equal(result.samples[mask], prompt[mask])
            assert result.metadata["nfe"] == 3
    handle.remove()
    assert len(calls) == 6
    assert all((state is None) == (i % 3 == 0) for i, state in enumerate(calls))
    assert all(state.shape == (1, 2, 8) for state in calls if state is not None)


@pytest.mark.parametrize("registers", [0, -1, 1.5, True])
def test_invalid_register_counts_rejected(registers):
    config = _config()
    config.algo.num_registers = registers
    with pytest.raises(ValueError, match="num_registers must be a positive integer"):
        CANDIRegisterLooped(config, Text8Tokenizer())


def test_sudoku_experiment_composes():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        config = compose(config_name="config", overrides=[
            "+experiment=sudoku_candi_register_looped",
        ])
    assert config.algo.name == "candi_register_looped"
    assert config.algo.backbone == "register_looped_cont_dit"
    assert config.algo.num_registers == 4
    assert config.sampling.predictor == "candi_looped"
    assert config.model.length == 180
