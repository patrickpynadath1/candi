from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- config resolvers
from algos.candi_looped import CANDILooped
from dataloader import Text8Tokenizer
from models.dit_cont import ContDIT
from samplers import SamplingRequest, create_sampler


def _config(*overrides):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        return compose(config_name="config", overrides=[
            "algo=candi_looped", "sampling.predictor=candi_looped",
            "model=tiny", "data=ptb", "training.ema=0",
            "model.length=4", "model.hidden_size=8", "model.cond_dim=8",
            "model.n_blocks=1", "model.n_heads=2", "model.dropout=0",
            *overrides,
        ])


def test_half_batch_recomputed_with_same_corruption_and_detached_hidden():
    model = CANDILooped(_config(), Text8Tokenizer())
    calls = []

    def record(_module, args, kwargs, output):
        output[0].retain_grad()
        output[1].retain_grad()
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
    first_kwargs, (first_logits, first_hidden) = calls[0]
    second_kwargs, (second_logits, _) = calls[1]
    selected = second_kwargs["example_ids"]
    assert len(selected) == 2 and len(selected.unique()) == 2
    for key in noisy:
        assert torch.equal(second_kwargs[key], first_kwargs[key][selected])
    assert torch.equal(second_kwargs["previous_hidden"], first_hidden[selected])
    assert not second_kwargs["previous_hidden"].requires_grad
    assert torch.count_nonzero(first_logits.grad[selected]) == 0
    remaining = torch.ones(4, dtype=torch.bool)
    remaining[selected] = False
    assert torch.count_nonzero(first_logits.grad[remaining]) > 0
    assert torch.count_nonzero(second_logits.grad) > 0
    assert torch.count_nonzero(model.backbone.feedback_projection.weight.grad) > 0


def test_first_pass_matches_base_and_feedback_changes_predictions():
    config = _config()
    model = CANDILooped(config, Text8Tokenizer()).eval()
    base = ContDIT(config, model.vocab_size).eval()
    base.load_state_dict({
        key: value for key, value in model.backbone.state_dict().items()
        if not key.startswith("feedback_")
    })
    noisy = model.q_xt(torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 1))
    with torch.no_grad():
        logits, hidden = model.backbone(**noisy, ret_embedding=True)
        assert torch.equal(logits, base(**noisy))
        assert not torch.allclose(
            logits, model.backbone(**noisy, previous_hidden=hidden)
        )
    with pytest.raises(ValueError, match="previous_hidden must match"):
        model.backbone(**noisy, previous_hidden=hidden[:, :-1])


@pytest.mark.parametrize("batch", [1, 3])
def test_odd_batches_round_randomly_and_selection_varies(batch):
    model = object.__new__(CANDILooped)
    torch.nn.Module.__init__(model)
    selections = []

    def forward_once(inputs, previous_hidden=None):
        if previous_hidden is not None:
            selections.append(tuple(inputs["xt"].flatten().tolist()))
        return inputs["xt"].float(), inputs["xt"].float()

    model._forward_once = forward_once
    inputs = {"xt": torch.arange(batch).reshape(batch, 1, 1)}
    sizes = set()
    for _ in range(40):
        before = len(selections)
        model._denoise_for_loss(inputs, True)
        sizes.add(len(selections[-1]) if len(selections) > before else 0)
    assert sizes == {batch // 2, batch // 2 + 1}
    if batch > 1:
        assert len(set(selections)) > 2


@pytest.mark.parametrize("loops", [1, 2, 3])
def test_sampling_counts_passes_preserves_prompt_and_resets_state(loops):
    model = CANDILooped(_config(f"algo.validation_loops={loops}"), Text8Tokenizer()).eval()
    calls = []
    outputs = []
    output_handle = model.backbone.register_forward_hook(
        lambda _module, args, output: outputs.append(output[1].detach().clone())
    )
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append(kwargs["previous_hidden"]),
        with_kwargs=True,
    )
    prompt = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[True, True, False, False]])
    sampler = create_sampler(model.config)
    sampler.validate_model(model)
    with torch.no_grad():
        for _ in range(2):
            result = sampler.sample(model, SamplingRequest(
                1, num_steps=3, prompt_tokens=prompt, prompt_mask=mask
            ))
            assert torch.equal(result.samples[mask], prompt[mask])
            assert result.metadata["nfe"] == 3
            assert result.metadata["diffusion_steps"] == 3
    handle.remove()
    output_handle.remove()
    assert len(calls) == 6
    assert all((state is None) == (i % 3 == 0) for i, state in enumerate(calls))
    for i, state in enumerate(calls):
        if i % 3:
            assert torch.equal(state, outputs[i - 1])
            assert not state.requires_grad


def test_validation_uses_configured_loops_without_half_batch_sampling():
    model = CANDILooped(_config(), Text8Tokenizer()).eval()
    calls = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append(kwargs["xt"].shape[0]),
        with_kwargs=True,
    )
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        loss = model._loss(tokens, torch.ones_like(tokens), train_mode=False).loss
    handle.remove()
    assert calls == [2, 2]
    assert torch.isfinite(loss)


@pytest.mark.parametrize("loops", [0, -1, 1.5, True])
def test_invalid_loop_count_rejected(loops):
    config = _config()
    config.algo.validation_loops = loops
    with pytest.raises(ValueError, match="positive integer"):
        CANDILooped(config, Text8Tokenizer())


def test_sudoku_experiment_composes():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base=None,
    ):
        config = compose(
            config_name="config", overrides=["+experiment=sudoku_candi_looped"]
        )
    assert config.algo.name == "candi_looped"
    assert config.algo.backbone == "looped_cont_dit"
    assert config.sampling.predictor == "candi_looped"
    assert config.model.length == 180


@pytest.mark.parametrize("algorithm", [
    "candi_looped", "candi_register_looped", "candi_softmax_looped",
])
@pytest.mark.parametrize("all_prompt", [False, True])
def test_looped_eb_greedy_preserves_prompt_and_counts_backbone_calls(
    algorithm, all_prompt, monkeypatch
):
    from algos import algo_registry

    def reject_sampling(*args, **kwargs):
        pytest.fail("greedy EB must not sample categorical token predictions")

    monkeypatch.setattr(
        "samplers.candi_entropy_bounded.sample_probabilities", reject_sampling
    )
    config = _config(
        f"algo={algorithm}",
        "sampling.predictor=candi_looped_entropy_bounded",
        "sampling.entropy_bound=0.1", "sampling.greedy_clean_estimate=true",
    )
    model = algo_registry.get(algorithm)(config, Text8Tokenizer()).eval()
    calls = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append(kwargs["previous_hidden"]),
        with_kwargs=True,
    )
    prompt = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    mask = torch.ones_like(prompt, dtype=torch.bool)
    if not all_prompt:
        mask[:, 2:] = False
    result = model.generate_samples(
        2, num_steps=4, prompt_tokens=prompt, prompt_mask=mask, return_result=True
    )
    handle.remove()
    assert torch.equal(result.samples[mask], prompt[mask])
    assert result.metadata["greedy_clean_estimate"]
    assert result.metadata["nfe"] == len(calls)
    assert result.metadata["nfe"] == result.metadata["diffusion_steps"]
    assert result.metadata["nfe_per_sample"] == [
        n for n in result.metadata["diffusion_steps_per_sample"]
    ]
    if all_prompt:
        assert not calls
    else:
        assert calls
        assert calls[0] is None
        assert all(h is not None for h in calls[1:])
