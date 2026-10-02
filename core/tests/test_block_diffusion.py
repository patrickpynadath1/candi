from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch

from core.modifiers.block_diffusion import BlockDiffusion, BlockSampler
from core.modifiers.block_diffusion.layout import make_layout
from core.samplers.ancestral import AncestralSampler
from core.training.module import TrainingModule


def configuration(*overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[1] / "new_configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "model.hidden_size=32", "model.n_heads=4", "model.n_blocks=1",
                "model.cond_dim=16", "model.dropout=0", "task.length=8",
                *overrides,
            ],
        )


def open_output(model):
    torch.nn.init.normal_(model.output_layer.linear.weight)
    torch.nn.init.normal_(model.output_layer.linear.bias)


def test_layout_has_no_clean_to_noisy_or_future_clean_leakage():
    layout = make_layout(6, 2, torch.device("cpu"))
    mask = layout.attention_mask[0, 0]
    assert not mask[:6, 6:].any()
    assert torch.equal(mask[:6, :6], torch.ones(6, 6, dtype=torch.bool).tril())
    # noisy block 1 sees clean block 0 and its own two noisy positions only
    assert mask[8].tolist() == [True, True, False, False, False, False,
                                False, False, True, True, False, False]


def test_full_block_matches_unmodified_model_and_future_clean_cannot_leak():
    torch.manual_seed(7)
    base = instantiate(configuration().algorithm).model.eval()
    open_output(base)
    reference = deepcopy(base)
    wrapped = BlockDiffusion(instantiate(configuration().algorithm), 8)
    wrapped.model.model.load_state_dict(base.state_dict())
    wrapped.eval()
    tokens = torch.randint(0, 12, (2, 8))
    noisy = torch.randint(0, 12, (2, 8))
    condition = torch.zeros(2)
    expected = reference(noisy, condition)
    with wrapped.model.clean_context(tokens):
        actual = wrapped.model(noisy, condition)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    wrapped.block_size = wrapped.model.block_size = 2
    changed = tokens.clone()
    changed[:, 2:] = (changed[:, 2:] + 1) % 12
    with wrapped.model.clean_context(tokens):
        first = wrapped.model(noisy, condition)
    with wrapped.model.clean_context(changed):
        second = wrapped.model(noisy, condition)
    torch.testing.assert_close(first[:, :2], second[:, :2])


def test_shifted_clean_loss_respects_target_mask_and_backpropagates():
    algorithm = BlockDiffusion(
        instantiate(configuration().algorithm), 2,
        clean_loss="autoregressive", clean_loss_weight=0.5,
    )
    open_output(algorithm.model.model)
    tokens = torch.randint(0, 12, (2, 8))
    mask = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]])
    result = algorithm.loss(tokens, mask)
    with algorithm.model.clean_context(tokens):
        algorithm.model(tokens, torch.zeros(len(tokens)))
        logits = algorithm.model.last_clean_logits
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none"
    )
    expected = (losses * mask[:, 1:]).sum() / mask[:, 1:].sum()
    torch.testing.assert_close(result["clean_loss"], expected)
    result["loss"].backward()
    assert algorithm.model.model.vocab_embed.embedding.grad is not None


def test_middle_block_visibility_and_prefix_kv_cache():
    torch.manual_seed(23)
    algorithm = BlockDiffusion(instantiate(configuration().algorithm), 2).eval()
    open_output(algorithm.model.model)
    clean = torch.randint(0, 12, (1, 8))
    noisy = torch.randint(0, 12, (1, 8))
    condition = torch.zeros(1)
    with algorithm.model.clean_context(clean):
        baseline = algorithm.model(noisy, condition)
    forbidden_clean = clean.clone()
    forbidden_clean[:, 2:] = (forbidden_clean[:, 2:] + 1) % 12
    forbidden_noisy = noisy.clone()
    forbidden_noisy[:, :2] = (forbidden_noisy[:, :2] + 1) % 12
    forbidden_noisy[:, 4:] = (forbidden_noisy[:, 4:] + 1) % 12
    with algorithm.model.clean_context(forbidden_clean):
        unchanged = algorithm.model(forbidden_noisy, condition)
    torch.testing.assert_close(baseline[:, 2:4], unchanged[:, 2:4])
    earlier = clean.clone()
    earlier[:, :2] = (earlier[:, :2] + 1) % 12
    with algorithm.model.clean_context(earlier):
        changed = algorithm.model(noisy, condition)
    assert not torch.equal(baseline[:, 2:4], changed[:, 2:4])

    prefix, active = clean[:, :4], noisy[:, 4:6]
    with algorithm.model.clean_context(prefix, offset=4):
        uncached = algorithm.model(active, condition)
    calls = 0
    def count(*unused):
        nonlocal calls
        calls += 1
    handle = algorithm.model.model.blocks[0].attn_qkv.register_forward_hook(count)
    cache = {}
    with algorithm.model.clean_context(prefix, offset=4, cache=cache):
        cached_first = algorithm.model(active, condition)
        cached_second = algorithm.model(active, condition)
    handle.remove()
    torch.testing.assert_close(cached_first, uncached, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cached_second, uncached, rtol=1e-5, atol=1e-6)
    assert calls == 3  # clean+noisy, then noisy only


def test_block_sampler_preserves_prompts_tail_and_reports_per_sample_nfe():
    torch.manual_seed(11)
    algorithm = BlockDiffusion(instantiate(configuration().algorithm), 3)
    sampler = BlockSampler(AncestralSampler(steps=1), 3)
    tokens = torch.randint(0, 12, (2, 8))
    prompt = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    output, metadata = sampler.sample(algorithm, tokens, prompt)
    assert output.shape == tokens.shape
    assert torch.equal(output[prompt], tokens[prompt])
    assert metadata["nfe_per_sample"] == [6, 6]
    assert metadata["nfe"] == 6


def test_mdlm_self_conditioning_combines_with_block_context():
    cfg = configuration(
        "self_conditioning.enabled=true", "self_conditioning.probability=1"
    )
    algorithm = BlockDiffusion(instantiate(cfg.algorithm), 2)
    tokens = torch.randint(0, 12, (2, 8))
    result = algorithm.loss(tokens, torch.ones_like(tokens))
    assert torch.isfinite(result["loss"])


def test_gaussian_block_loss_and_cached_flag_have_same_generation_semantics():
    cfg = configuration("+experiment=gaussian_diffusion_sudoku")
    first = BlockDiffusion(instantiate(cfg.algorithm), 4)
    second = BlockDiffusion(instantiate(cfg.algorithm), 4)
    second.load_state_dict(first.state_dict())
    tokens = torch.randint(0, 12, (1, 8))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    torch.manual_seed(19)
    cached, _ = first.generate(
        BlockSampler(__import__("core.samplers.euler", fromlist=["EulerSampler"]).EulerSampler(steps=1), 4, True),
        tokens, mask,
    )
    torch.manual_seed(19)
    uncached, _ = second.generate(
        BlockSampler(__import__("core.samplers.euler", fromlist=["EulerSampler"]).EulerSampler(steps=1), 4, False),
        tokens, mask,
    )
    assert torch.equal(cached, uncached)


def test_training_module_assembly_and_state_roundtrip():
    cfg = configuration(
        "output_dir=/tmp/block_diffusion_test",
        "data.cache_dir=/tmp/block_diffusion_data",
        "block_diffusion.enabled=true",
        "block_diffusion.block_size=2",
    )
    first = TrainingModule(cfg)
    assert isinstance(first.algorithm, BlockDiffusion)
    state = first.state_dict()
    second = TrainingModule(cfg)
    second.load_state_dict(state)
    assert state.keys() == second.state_dict().keys()
    for key, value in state.items():
        torch.testing.assert_close(value, second.state_dict()[key])
