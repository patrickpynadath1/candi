"""MDLM shares the composed loop without changing its training/checkpoint API."""

from unittest.mock import patch
import pytest
import torch
from hydra.utils import instantiate

from core.modifiers.block_diffusion import BlockDiffusion, BlockSampler
from core.samplers.forward_resampling import ForwardResamplingSampler
from core.tests.test_block_diffusion import configuration, open_output


@pytest.mark.parametrize("source", ["none", "hidden", "posterior_mean"])
@pytest.mark.parametrize("greedy", [False, True])
def test_block_renoising_preserves_context_feedback_and_budget(source, greedy):
    cfg = configuration(
        f"self_conditioning.enabled={str(source != 'none').lower()}",
        f"self_conditioning.source={source if source != 'none' else 'hidden'}",
        "self_conditioning.probability=1",
    )
    algorithm = BlockDiffusion(instantiate(cfg.algorithm), 3).eval()
    open_output(algorithm.model.model)
    tokens = torch.arange(16).reshape(2, 8) % 9
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, 0] = True
    before = set(algorithm.state_dict())
    results = []
    for cache in (False, True):
        sampler = BlockSampler(
            ForwardResamplingSampler(steps=2, greedy_clean_estimate=greedy),
            3, cache_clean_prefix=cache,
        )
        process = algorithm.algorithm.process
        with patch.object(process, "corrupt_state", wraps=process.corrupt_state) as corrupt:
            torch.manual_seed(81)
            output, info = sampler.sample(algorithm, tokens, prompt)
        assert corrupt.call_count == 6  # two transitions in each of three blocks
        assert torch.equal(output[prompt], tokens[prompt])
        assert (output != process.mask_index).all()
        assert info["nfe"] == 9
        assert info["nfe_per_sample"] == [9, 9]
        results.append(output)
    assert torch.equal(*results)
    assert set(algorithm.state_dict()) == before


def test_adapter_preserves_predictions_and_forward_kernel():
    algorithm = instantiate(configuration("algorithm.time_conditioning=true").algorithm).eval()
    adapter = algorithm.sampler_adapter()
    tokens = torch.arange(16).reshape(2, 8) % 9
    time = torch.tensor([0.3, 0.8])
    points = adapter.points(time)
    keep = points[0].coefficients[0]
    torch.testing.assert_close(
        adapter.sampling_prediction((tokens,), points, time),
        algorithm.sampling_prediction(tokens, keep),
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    torch.manual_seed(42)
    expected = algorithm.process.corrupt(tokens, keep, mask)
    torch.manual_seed(42)
    actual = adapter.views[0].process.corrupt_state(tokens, points[0].coefficients, mask)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("encoder_layers", [0, 1, 2])
@pytest.mark.parametrize("greedy", [False, True])
def test_encoder_decoder_renoising(encoder_layers, greedy):
    from core.sweeps.configuration import compose
    from core.training.module import TrainingModule
    config = compose("e2d2", [
        "model.hidden_size=32", "model.n_heads=4", "model.n_blocks=1",
        "model.cond_dim=16", "model.dropout=0", "task.length=8",
        "block_diffusion.block_size=3",
        f"encoder_decoder.encoder_layers={encoder_layers}",
    ])
    algorithm = TrainingModule(config).algorithm.eval()
    tokens = torch.arange(16).reshape(2, 8) % 9
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, :2] = True
    outputs = []
    for cache in (False, True):
        sampler = BlockSampler(
            ForwardResamplingSampler(steps=2, greedy_clean_estimate=greedy),
            3, cache_clean_prefix=cache,
        )
        torch.manual_seed(42)
        output, info = sampler.sample(algorithm, tokens, prompt)
        assert torch.equal(output[prompt], tokens[prompt])
        assert info["nfe_per_sample"] == [9, 9]
        outputs.append(output)
    assert torch.equal(*outputs)
