import pytest
import torch

from core.sweeps.configuration import compose
from core.training.module import TrainingModule


def model(checkpoint=False, encoder_layers=2):
    config = compose(
        "e2d2",
        [
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=2",
            "model.dropout=0",
            "block_diffusion.block_size=2",
            f"encoder_decoder.encoder_layers={encoder_layers}",
        ],
    )
    wrapped = TrainingModule(config).algorithm.model
    torch.nn.init.normal_(wrapped.output_layer.linear.weight, std=0.1)
    wrapped.model.activation_checkpointing = checkpoint
    wrapped.model.encoder_execution.activation_checkpointing = checkpoint
    return wrapped


@pytest.mark.parametrize("encoder_layers", [0, 2])
def test_no_current_or_future_answer_leakage_and_context_has_effect(encoder_layers):
    torch.manual_seed(14)
    m = model(encoder_layers=encoder_layers).eval()
    clean = torch.randint(0, 12, (2, 8))
    noisy = torch.randint(0, 12, (2, 8))
    t = torch.zeros(2)

    def predict(x):
        with m.clean_context(x):
            return m(noisy, t)

    baseline = predict(clean)
    changed = clean.clone()
    changed[:, 2:] = (changed[:, 2:] + 1) % 12
    torch.testing.assert_close(baseline[:, :4], predict(changed)[:, :4])
    changed = clean.clone()
    changed[:, :2] = (changed[:, :2] + 1) % 12
    assert not torch.allclose(baseline[:, 2:4], predict(changed)[:, 2:4])


@pytest.mark.parametrize("encoder_layers", [0, 2])
@pytest.mark.parametrize("start", [0, 2, 6])
@torch.no_grad()
def test_parallel_matches_blockwise_and_cached(start, encoder_layers):
    m = model(encoder_layers=encoder_layers).eval()
    clean = torch.randint(0, 12, (2, 8))
    noisy = torch.randint(0, 12, (2, 8))
    t = torch.zeros(2)
    with m.clean_context(clean):
        expected = m(noisy, t)[:, start : start + 2]
    cache = {}
    with m.clean_context(clean[:, :start], offset=start, cache=cache):
        actual = m(noisy[:, start : start + 2], t)
        cached = m(noisy[:, start : start + 2], t)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, cached)
    assert "decoder_context_kv" in cache


def test_checkpoint_gradients_and_encoder_receives_training_signal():
    eager = model().train()
    checked = model().train()
    checked.load_state_dict(eager.state_dict())
    checked.model.activation_checkpointing = True
    checked.model.encoder_execution.activation_checkpointing = True
    clean = torch.randint(0, 12, (2, 8))
    noisy = torch.randint(0, 12, (2, 8))
    t = torch.zeros(2)
    for m in (eager, checked):
        with m.clean_context(clean):
            m(noisy, t)[:, 2:].square().mean().backward()
    for (name, a), (_, b) in zip(eager.named_parameters(), checked.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
    assert eager.model.encoder[0].attn_qkv.weight.grad.norm() > 0


@pytest.mark.parametrize(
    "recipe", ["e2d2", "blockgen_uniform", "gaussian_diffusion_sudoku"]
)
@pytest.mark.parametrize("encoder_layers", [0, 2])
def test_wrapper_composes_with_corruption_and_trains(recipe, encoder_layers):
    from omegaconf import OmegaConf

    config = compose(
        recipe,
        [
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=2",
            "model.dropout=0",
            "task.length=8",
            "block_diffusion.enabled=true",
            "block_diffusion.block_size=2",
        ],
    )
    config.encoder_decoder = OmegaConf.create(
        {
            "_target_": "core.modifiers.encoder_decoder.EncoderDecoderModel",
            "encoder_layers": encoder_layers,
        }
    )
    algo = TrainingModule(config).algorithm
    tokens = torch.randint(0, 9, (2, 8))
    loss = algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool), training=True)[
        "loss"
    ]
    assert torch.isfinite(loss)
    loss.backward()


def test_depth_allocations_match_parameter_count():
    counts = []
    for enc, dec in [(10, 2), (6, 6), (2, 10)]:
        config = compose(
            "e2d2",
            [
                "model.hidden_size=32",
                "model.n_heads=4",
                f"model.n_blocks={dec}",
                f"encoder_decoder.encoder_layers={enc}",
            ],
        )
        counts.append(
            sum(p.numel() for p in TrainingModule(config).algorithm.parameters())
        )
    assert len(set(counts)) == 1


@pytest.mark.parametrize("recipe", ["bd3lm", "e2d2"])
@pytest.mark.parametrize("causal", [False, True])
@torch.no_grad()
def test_incremental_prefix_cache_matches_recompute_and_only_encodes_new_tokens(
    recipe, causal
):
    config = compose(
        recipe,
        [
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=2",
            "model.dropout=0",
            "block_diffusion.block_size=2",
            f"block_diffusion.x0_causal={str(causal).lower()}",
        ],
    )
    if recipe == "e2d2":
        config.encoder_decoder.encoder_layers = 2
    m = TrainingModule(config).algorithm.model.eval()
    torch.nn.init.normal_(m.output_layer.linear.weight, std=0.1)
    clean = torch.randint(0, 12, (2, 8))
    noisy = torch.randint(0, 12, (2, 8))
    t = torch.zeros(2)
    lengths = []
    encoder = m.model.encoder if recipe == "e2d2" else m.model.blocks
    cache = {}
    for start in (0, 2, 4, 6):
        with m.clean_context(clean[:, :start], offset=start):
            expected = m(noisy[:, start : start + 2], t)

        def record(_module, args, kwargs):
            if kwargs.get("return_key_value", False):
                lengths.append(args[0].shape[1])

        hook = encoder[0].register_forward_pre_hook(record, with_kwargs=True)
        with m.clean_context(clean[:, :start], offset=start, cache=cache):
            actual = m(noisy[:, start : start + 2], t)
            repeated = m(noisy[:, start : start + 2], t)
        hook.remove()
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual, repeated)
    assert lengths == [2, 2, 2]
    assert cache["clean_length"] == 6
    changed = clean[:, :6].clone()
    changed[:, 0] = (changed[:, 0] + 1) % 12
    with m.clean_context(changed, offset=6):
        expected = m(noisy[:, 6:], t)
    with m.clean_context(changed, offset=6, cache=cache):
        actual = m(noisy[:, 6:], t)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("recipe", ["bd3lm", "e2d2"])
@torch.no_grad()
def test_extending_incomplete_clean_block_invalidates_cache(recipe):
    config = compose(
        recipe,
        [
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=2",
            "model.dropout=0",
            "block_diffusion.block_size=2",
        ],
    )
    m = TrainingModule(config).algorithm.model.eval()
    torch.nn.init.normal_(m.output_layer.linear.weight, std=0.1)
    x = torch.randint(0, 12, (1, 6))
    t = torch.zeros(1)
    cache = {}
    with m.clean_context(x[:, :1], offset=4, cache=cache):
        m(x[:, 4:], t)
    with m.clean_context(x[:, :4], offset=4, cache=cache):
        actual = m(x[:, 4:], t)
    with m.clean_context(x[:, :4], offset=4):
        expected = m(x[:, 4:], t)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("recipe", ["bd3lm", "e2d2"])
@torch.no_grad()
def test_generation_reuses_prefix_cache_but_starts_fresh_each_time(recipe):
    from core.modifiers.block_diffusion import BlockSampler
    from core.samplers.ancestral import AncestralSampler

    config = compose(
        recipe,
        [
            "model.hidden_size=32",
            "model.n_heads=4",
            "model.n_blocks=2",
            "model.dropout=0",
            "block_diffusion.block_size=2",
            "task.length=8",
        ],
    )
    algo = TrainingModule(config).algorithm.eval()
    encoder = algo.model.model.encoder if recipe == "e2d2" else algo.model.model.blocks
    encoded = []

    def record(_module, args, kwargs):
        if kwargs.get("return_key_value", False):
            encoded.append(args[0].shape[1])

    handle = encoder[0].register_forward_pre_hook(record, with_kwargs=True)
    sampler = BlockSampler(AncestralSampler(steps=2), block_size=2)
    prompt = torch.zeros(1, 8, dtype=torch.long)
    for _ in range(2):
        result, _ = sampler.sample(
            algo, prompt, torch.zeros_like(prompt, dtype=torch.bool)
        )
        assert result.shape == prompt.shape
    handle.remove()
    assert encoded == [2, 2, 2, 2, 2, 2]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_raw_context_projects_embeddings_and_trains_them(checkpoint, monkeypatch):
    m = model(checkpoint=checkpoint, encoder_layers=0).train()
    assert len(m.model.encoder) == 0
    clean = torch.ones(2, 4, dtype=torch.long)
    noisy = torch.full((2, 2), 2, dtype=torch.long)
    seen = []
    block = m.model.model.blocks[0]
    project = block.project_context_key_value

    def record(context, rotary):
        seen.append(context.detach().clone())
        return project(context, rotary)

    monkeypatch.setattr(block, "project_context_key_value", record)
    with m.clean_context(clean, offset=4):
        m(noisy, torch.zeros(2)).square().mean().backward()
    torch.testing.assert_close(seen[0], m.vocab_embed(clean))
    # Token1 occurs only in clean context, proving gradients through that path.
    assert m.vocab_embed.embedding.grad[1].norm() > 0


@torch.no_grad()
def test_raw_context_cache_only_projects_new_tokens(monkeypatch):
    m = model(encoder_layers=0).eval()
    block = m.model.model.blocks[0]
    project = block.project_context_key_value
    lengths = []

    def record(context, rotary):
        lengths.append(context.shape[1])
        return project(context, rotary)

    clean = torch.randint(0, 12, (2, 8))
    noisy = torch.randint(0, 12, (2, 8))
    cache = {}
    for start in (0, 2, 4, 6):
        with m.clean_context(clean[:, :start], offset=start):
            expected = m(noisy[:, start : start + 2], torch.zeros(2))
        with monkeypatch.context() as patch:
            patch.setattr(block, "project_context_key_value", record)
            with m.clean_context(clean[:, :start], offset=start, cache=cache):
                actual = m(noisy[:, start : start + 2], torch.zeros(2))
                repeated = m(noisy[:, start : start + 2], torch.zeros(2))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual, repeated)
    assert lengths == [2, 2, 2]
    assert cache["encoder_kv"] == []
