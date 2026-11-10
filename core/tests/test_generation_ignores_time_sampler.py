"""Generation never consults the training time sampler, for every preset.

The training time sampler decides how often training visits each time; it has no
role at inference. A learned time map that generation reads (adaptive views,
simplex ``time_grid=adaptive``) is an information-path component
(``algorithm.time_map``). For each preset in ``configs/experiment`` this trains
one step, generates, then replaces every training time sampler in the assembly
with an object that raises on any access and generates again: the tokens must be
identical.
"""


import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from core.sweeps.configuration import compose
from core.tests.helpers import ROOT
from core.training.module import build_algorithm

PRESETS = sorted(path.stem for path in (ROOT / "configs/experiment").glob("*.yaml"))
SHRINK = {"hidden_size": 32, "n_heads": 4, "n_blocks": 1, "cond_dim": 16, "dropout": 0.0}


class Untouchable:
    def __getattr__(self, name):
        raise AssertionError(f"generation consulted the training time sampler ({name})")


def shrink(node):
    """Shrink every network width/depth setting in place."""
    if OmegaConf.is_dict(node):
        for key in list(node.keys()):
            value = node._get_node(key)
            if key in SHRINK and not OmegaConf.is_interpolation(node, key):
                if isinstance(node[key], (int, float)) and not isinstance(node[key], bool):
                    node[key] = SHRINK[key]
            elif OmegaConf.is_dict(value) or OmegaConf.is_list(value):
                shrink(value)
    elif OmegaConf.is_list(node):
        for item in node:
            if OmegaConf.is_dict(item) or OmegaConf.is_list(item):
                shrink(item)


def tiny(preset):
    config = compose(preset)
    try:
        instantiate(config.sampler)
    except Exception:  # noqa: BLE001
        pytest.skip("base preset without a usable sampler of its own")
    if config.get("strategy") is not None or config.trainer.strategy not in ("auto", None):
        pytest.skip("multi-GPU preset")
    with open_dict(config):
        shrink(config.model)
    # Long text contexts (e.g. 1024 tokens) only cost CPU time here.
    if "length" in config.task and int(config.task.length) > 128:
        config.task.length = 128
    # Fewer generation steps, except for samplers that stop on their own
    # (entropy-bounded samplers must reach completion within their budget).
    target = str(config.sampler.get("_target_", ""))
    if "steps" in config.sampler.keys() and "entropy_bounded" not in target:
        config.sampler.steps = 3
    return config


def replace_time_samplers(algorithm):
    """Swap every training time sampler in the (possibly wrapped) assembly."""
    replaced = 0
    seen = set()
    stack = [algorithm]
    while stack:
        component = stack.pop()
        if id(component) in seen:
            continue
        seen.add(id(component))
        if "time_sampler" in vars(component) or "time_sampler" in getattr(component, "_modules", {}):
            component._modules.pop("time_sampler", None)
            component.__dict__["time_sampler"] = Untouchable()
            replaced += 1
        samplers = getattr(component, "time_samplers", None)
        if isinstance(samplers, torch.nn.ModuleDict):
            for key in list(samplers.keys()):
                samplers._modules[key] = None
            component.__dict__["time_samplers"] = Untouchable()
            component._modules.pop("time_samplers", None)
        for child in getattr(component, "_modules", {}).values():
            if child is not None:
                stack.append(child)
        inner = vars(component).get("algorithm")
        if inner is not None:
            stack.append(inner)
    return replaced


def batch(config, algorithm):
    vocab = int(config.task.vocab_size)
    length = int(getattr(algorithm, "length", config.task.length))
    torch.manual_seed(0)
    tokens = torch.randint(0, max(vocab - 2, 1), (2, length))
    # Prompts never contain an absorbing mask token inside the vocabulary.
    mask = OmegaConf.select(config, "token_space.mask_index", default=None)
    if mask is not None and mask < vocab:
        tokens[tokens == mask] = (mask + 1) % max(vocab - 2, 1)
    loss_mask = torch.ones_like(tokens, dtype=torch.bool)
    loss_mask[:, : max(1, length // 4)] = False
    return tokens, loss_mask


@pytest.mark.parametrize("preset", PRESETS)
def test_generation_never_consults_the_training_time_sampler(preset):
    config = tiny(preset)
    algorithm = build_algorithm(config)
    if not any(
        "time_sampler" in vars(m) or "time_sampler" in m._modules for m in algorithm.modules()
    ) and getattr(algorithm, "time_sampler", None) is None:
        pytest.skip("no training time sampler (autoregressive recipe)")
    tokens, loss_mask = batch(config, algorithm)
    options = (
        {"valid_mask": torch.ones_like(loss_mask)}
        if getattr(algorithm, "supports_variable_length", False)
        else {}
    )
    algorithm.train()
    algorithm.loss(tokens, loss_mask, corruption_mask=loss_mask, step=0, training=True,
                   **options)["loss"].backward()
    algorithm.eval()
    prompt = ~loss_mask
    torch.manual_seed(3)
    expected, _ = algorithm.generate(instantiate(config.sampler), tokens, prompt)
    replace_time_samplers(algorithm)
    torch.manual_seed(3)
    result, _ = algorithm.generate(instantiate(config.sampler), tokens, prompt)
    assert torch.equal(result, expected)
