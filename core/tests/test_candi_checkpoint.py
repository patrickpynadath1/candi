"""Regression tests for positional EMA import and strict parameter coverage."""

from collections import OrderedDict

import pytest
import torch
from torch import nn

from core.training.candi_checkpoint import legacy_candi_state, map_legacy_candi_state


def checkpoint():
    table = torch.zeros(3, 4)
    state = OrderedDict([
        ("backbone.vocab_embed.embedding", table),
        ("backbone.sigma_map.weight", torch.zeros(2, 4)),
        ("backbone.rotary_emb.inv_freq", torch.ones(2)),
        ("backbone.precond.embed_table.embedding", table),
        ("backbone.output_layer.weight", torch.zeros(3, 4)),
    ])
    return {"state_dict": state, "ema": {"shadow_params": [
        torch.full((3, 4), 1.), torch.full((2, 4), 2.), torch.full((3, 4), 3.),
    ]}}


def test_ema_skips_buffer_and_shared_parameter_alias():
    state = legacy_candi_state(checkpoint(), use_ema=True)
    assert state["backbone.vocab_embed.embedding"].eq(1).all()
    assert state["backbone.sigma_map.weight"].eq(2).all()
    assert state["backbone.output_layer.weight"].eq(3).all()
    assert state["backbone.precond.embed_table.embedding"] is state["backbone.vocab_embed.embedding"]
    assert state["backbone.rotary_emb.inv_freq"].eq(1).all()


def test_ema_missing_or_wrong_shape_fails_explicitly():
    saved = checkpoint()
    del saved["ema"]
    with pytest.raises(ValueError, match="no EMA"):
        legacy_candi_state(saved, use_ema=True)
    saved = checkpoint()
    saved["ema"]["shadow_params"][1] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        legacy_candi_state(saved, use_ema=True)


ORIGINAL_NAMES = [
    ("model.input.views.0.representation.embedding", "backbone.vocab_embed.embedding"),
    ("model.time.embedder.", "backbone.sigma_map."),
    ("model.rotary_emb.", "backbone.rotary_emb."),
    ("model.blocks.", "backbone.blocks."),
    ("model.output.layer.", "backbone.output_layer."),
]


def candi_module(seed, n_blocks=1):
    from core.sweeps import compose
    from core.training.module import TrainingModule

    config = compose("candi_hybrid", {
        "model.hidden_size": 32, "model.n_heads": 2, "model.cond_dim": 16, "model.dropout": 0, "model.n_blocks": n_blocks, "model.views.0.representation.hidden_size": 32,
        "training.ema": 0.9, "sampler.steps": 40,
    })
    torch.manual_seed(seed)
    return TrainingModule(config)


def original_checkpoint(module):
    """The original CANDI file layout of ``module``'s weights, with an EMA."""
    state = module.algorithm.state_dict()
    original = OrderedDict()
    for new, old in ORIGINAL_NAMES:
        for key, value in state.items():
            if key.startswith(new):
                original[old + key[len(new):]] = value.clone()
    original["backbone.precond.embed_table.embedding"] = original["backbone.vocab_embed.embedding"]
    names = [k for k in original if k not in {"backbone.rotary_emb.inv_freq", "backbone.precond.embed_table.embedding"}]
    ema = {name: original[name] + 0.5 for name in names}
    return {
        "state_dict": original,
        "ema": {"decay": 0.9, "num_updates": 11, "shadow_params": [ema[n] for n in names]},
        "global_step": 6,
    }, ema


def test_original_candi_checkpoint_imports_through_the_converter(tmp_path):
    from core.training.checkpoints import load_weights

    source = candi_module(0)
    with torch.no_grad():
        for parameter in source.algorithm.parameters():
            parameter.add_(torch.randn_like(parameter))
    saved, ema = original_checkpoint(source)
    path = tmp_path / "candi.ckpt"
    torch.save(saved, path)
    target = candi_module(42)
    assert load_weights(target, path, legacy="candi") == 6
    for (name, a), b in zip(source.algorithm.named_parameters(), target.algorithm.parameters()):
        assert torch.equal(a, b), name
    for key, value in source.algorithm.state_dict().items():
        assert torch.equal(value, target.algorithm.state_dict()[key]), key
    expected = {}
    for new, old in ORIGINAL_NAMES:
        for name, _ in source.algorithm.named_parameters():
            if name.startswith(new):
                expected[name] = ema[old + name[len(new):]]
    shadows = dict(zip([n for n, _ in target.algorithm.named_parameters()], target.ema.shadow_params))
    assert shadows.keys() == expected.keys()
    assert all(torch.equal(shadows[n], expected[n]) for n in expected)
    assert target.ema.num_updates == 11
    # Same samples from the imported weights.
    from hydra.utils import instantiate

    sampler = instantiate(source.config.sampler)
    prompt = torch.randint(0, source.config.task.vocab_size, (2, source.config.task.length))
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :5] = True
    source.algorithm.eval()
    target.algorithm.eval()
    torch.manual_seed(3)
    expected_tokens, expected_info = source.algorithm.generate(sampler, prompt, fixed)
    torch.manual_seed(3)
    tokens, info = target.algorithm.generate(sampler, prompt, fixed)
    assert torch.equal(expected_tokens, tokens)
    assert expected_info["nfe_per_sample"] == info["nfe_per_sample"]


def test_mapper_rejects_unmapped_parameters_and_shape_mismatch():
    module = candi_module(0)
    saved, _ = original_checkpoint(module)
    state = legacy_candi_state(saved)
    mapped = map_legacy_candi_state(state, module.algorithm)
    assert torch.equal(mapped["model.input.views.1.representation.embedding"], state["backbone.vocab_embed.embedding"])
    del state["backbone.output_layer.linear.weight"]
    with pytest.raises(ValueError, match="unmapped trainable"):
        map_legacy_candi_state(state, module.algorithm)
    state = legacy_candi_state(saved)
    state["backbone.sigma_map.mlp.0.weight"] = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        map_legacy_candi_state(state, module.algorithm)
    with pytest.raises(TypeError, match="view-input ModularDiT"):
        map_legacy_candi_state(state, nn.Linear(2, 2))
