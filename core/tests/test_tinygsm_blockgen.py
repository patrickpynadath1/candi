"""BlockGen recipe construction and the semantics needed by its two losses."""

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch

from core.modifiers.block_diffusion.layout import make_layout
from core.training.module import TrainingModule


def recipe(kind, *overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        return compose(config_name="config", overrides=[
            f"+experiment=tinygsm_blockgen_{kind}",
            "output_dir=/tmp/blockgen-test", "data.cache_dir=/tmp/blockgen-data",
            *overrides,
        ])


@pytest.mark.parametrize("kind", ["masked", "uniform"])
def test_recipe_loss_backward_and_time_independence(kind):
    cfg = recipe(kind)
    assert cfg.loader.global_batch_size == 512
    partitions = cfg.trainer.devices * cfg.trainer.accumulate_grad_batches
    assert cfg.loader.global_batch_size // partitions == 8
    assert cfg.data.train_on_pad and not cfg.data.train_on_prompt
    assert cfg.data.separator == "\\n"
    assert cfg.task.length == 512 and cfg.trainer.max_steps == 250000
    # Exercise the real algorithm/config assembly without downloading TinyGSM.
    cfg.model.hidden_size, cfg.model.cond_dim = 32, 16
    cfg.model.n_blocks, cfg.model.n_heads, cfg.model.dropout = 1, 4, 0
    cfg.task.vocab_size, cfg.task.length = 17, 8
    cfg.block_diffusion.block_size = 4
    module = TrainingModule(cfg)
    algorithm = module.algorithm
    tokens = torch.randint(17, (2, 8))
    mask = torch.ones_like(tokens)
    mask[:, :2] = 0
    result = algorithm.loss(tokens, mask)
    assert result["num_tokens"] == 12
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert algorithm.model.output_layer.linear.weight.grad.abs().sum() > 0

    seen = []
    handle = algorithm.model.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[1].detach().clone())
    )
    with algorithm.model.clean_context(tokens):
        algorithm.training_prediction(tokens, torch.tensor([0.2, 0.8]))
        algorithm.sampling_prediction(tokens, torch.tensor([0.3, 0.9]))
    handle.remove()
    assert len(seen) == 2 and all(torch.equal(t, torch.zeros(2)) for t in seen)


def test_masked_ce_has_no_time_weight_and_counts_all_eligible_positions():
    cfg = recipe("masked", "model.hidden_size=32", "model.cond_dim=16",
                 "model.n_blocks=1", "model.n_heads=4", "model.dropout=0",
                 "task.vocab_size=3", "task.length=4")
    algorithm = TrainingModule(cfg).algorithm.algorithm
    tokens = torch.tensor([[0, 1, 2, 2]])
    eligible = torch.tensor([[0, 1, 1, 1]])  # final position represents PAD
    log_probs = torch.tensor([[[0., -10., -10., -10.],
                              [-10., -2., -10., -10.],
                              [-10., -10., 0., -10.],
                              [-10., -10., -4., -10.]]])
    algorithm.training_prediction = lambda *_args: log_probs
    # Two different noise levels must produce the same unweighted objective.
    for time in (0.1, 0.9):
        algorithm.time_sampler.sample = lambda *_args, **_kwargs: torch.tensor([time])
        result = algorithm.loss(tokens, eligible)
        torch.testing.assert_close(result["loss"], torch.tensor(2.0))


def test_block_causal_clean_stream_sees_own_block_but_not_future_blocks():
    mask = make_layout(8, 4, torch.device("cpu"), x0_causal=False).attention_mask[0, 0]
    assert mask[0, :4].all() and not mask[0, 4:].any()
    assert mask[4, :8].all() and not mask[4, 8:].any()
    # Noisy second block sees clean first block and noisy second block only.
    assert mask[12].tolist() == [True] * 4 + [False] * 8 + [True] * 4
