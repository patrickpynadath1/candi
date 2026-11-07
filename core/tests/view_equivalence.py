"""Shared checks that a family moved onto the view path is bit-identical.

A family test (core/tests/test_view_equivalence_<family>.py) builds the frozen
standalone class (core/tests/reference_algorithms/<family>.py) and the current
class from the same config and seed, then requires torch.equal on
initialization, loss, gradients, samples and NFE.
"""

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from core.sweeps import compose

SMALL = {
    "model.hidden_size": 32,
    "model.n_blocks": 2,
    "model.n_heads": 2,
    "model.cond_dim": 16,
    "model.dropout": 0,
}


def build(experiment, target, self_conditioning=None, seed=0, overrides=None):
    """(config, algorithm) for ``experiment``; ``target`` replaces the class."""
    options = {**SMALL, **(overrides or {})}
    if self_conditioning is not None:
        options["self_conditioning.enabled"] = self_conditioning
    cfg = compose(experiment, options)
    if target is not None:
        OmegaConf.update(cfg, "algorithm._target_", target)
    torch.manual_seed(seed)
    return cfg, instantiate(cfg.algorithm)


def batch(cfg, size=4, prompt=5):
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.task.vocab_size, (size, cfg.task.length))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :prompt] = False
    return tokens, mask


def _pairs(reference, current, ordered):
    """Corresponding parameters: by position, or matched by shape and value."""
    old, new = list(reference.parameters()), list(current.parameters())
    if ordered:
        return list(zip(old, new))
    pairs, remaining = [], list(new)
    for a in old:
        index = next(
            (i for i, b in enumerate(remaining) if a.shape == b.shape and torch.equal(a, b)), None
        )
        assert index is not None, tuple(a.shape)
        pairs.append((a, remaining.pop(index)))
    return pairs


def assert_same_parameters(reference, current, ordered=True):
    """Identical parameter values; in identical order unless ``ordered=False``
    (then matched as multisets of (shape, values), for networks whose legacy
    class held bare root parameters, e.g. ContinuousDiT)."""
    assert len(list(reference.parameters())) == len(list(current.parameters()))
    assert all(a.shape == b.shape and torch.equal(a, b) for a, b in _pairs(reference, current, ordered))


def assert_training_identical(reference, current, tokens, mask, seed=7, ordered=True, **loss_options):
    """Same loss and every gradient (paired by value) for one training step."""
    assert_same_parameters(reference, current, ordered=ordered)
    results = []
    for algorithm in (reference, current):
        algorithm.train()
        torch.manual_seed(seed)
        result = algorithm.loss(tokens, mask, training=True, **loss_options)
        result["loss"].backward()
        results.append(result)
    assert torch.equal(results[0]["loss"], results[1]["loss"])
    for a, b in _pairs(reference, current, ordered):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
    return results


def assert_sampling_identical(reference, current, prompt, fixed, samplers, seed=11):
    """Same tokens and per-sample NFE for every sampler config (dicts)."""
    reference.eval()
    current.eval()
    for options in samplers:
        sampler = instantiate(options)
        torch.manual_seed(seed)
        expected, expected_info = reference.generate(sampler, prompt, fixed)
        torch.manual_seed(seed)
        actual, actual_info = current.generate(sampler, prompt, fixed)
        assert torch.equal(expected, actual), options
        assert expected_info["nfe_per_sample"] == actual_info["nfe_per_sample"], options
