"""Structural invariants, insertion laws, and train/sample integration."""

from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.sequence_edits import SequenceEdit
from core.modifiers.variable_length.birth import ForwardInsertion, MaskInsertion
from core.modifiers.variable_length.training import VariableLengthDiffusion
from core.modifiers.hybrid.views import point
from core.noise_schedules.insertion import PowerInsertionSchedule
from core.processes.deletion import DeletionProcess
from core.samplers.insertion import InsertionSampler
from core.tests.test_hybrid_diffusion import make_algorithm


def assembly(feedback=False, *, continuous=False):
    inner = make_algorithm(self_conditioning=feedback)
    # Use one assembly contract for single-view and hybrid models.
    from core.modifiers.hybrid.model import HybridModel
    from core.models.dit import DiT

    views = [inner.views[1]] if continuous else list(inner.views)
    inner.model = HybridModel(
        DiT(5, 16, 8, 1, 4, dropout=0, adaLN=True), views, self_conditioning=feedback
    )
    policies = (
        [ForwardInsertion("mean")]
        if continuous
        else [MaskInsertion(), ForwardInsertion("mean")]
    )
    return VariableLengthDiffusion(inner, 5, policies)


def test_shared_edit_preserves_order_dtype_gradients_and_empty_rows():
    old = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    edit = SequenceEdit.insertion(
        torch.tensor([2, 0]), torch.tensor([[2, 1, 2], [3, 0, 0]])
    )
    born = torch.tensor([[1.0, 2.0, 3.0], [7.0, 0.0, 0.0]], requires_grad=True)
    result = edit.apply(old, born)
    assert result.tolist() == [[1, 1, 10, 2, 20, 3, 3], [7, 7, 7, 0, 0, 0, 0]]
    assert edit.apply(torch.ones_like(old, dtype=torch.bool)).dtype == torch.bool
    result.sum().backward()
    torch.testing.assert_close(old.grad, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    torch.testing.assert_close(
        born.grad, torch.tensor([[2.0, 1.0, 2.0], [3.0, 0.0, 0.0]])
    )
    compact = SequenceEdit.deletion(torch.tensor([[False, True], [False, False]]))
    assert compact.apply(old.detach()).tolist() == [[20], [0]]
    assert compact.valid.tolist() == [[True], [False]]


def test_edit_rejects_counts_in_padding_gaps():
    with pytest.raises(ValueError, match="padding gaps"):
        SequenceEdit.insertion(torch.tensor([0]), torch.tensor([[0, 1]]))


def test_deletion_alignment_targets_and_protected_positions():
    class FixedBirths(PowerInsertionSchedule):
        def birth_times(self, shape, device):
            return torch.tensor([[0.1, 0.8, 0.2, 0.9, 0.1]], device=device)

    valid = torch.ones(1, 5, dtype=torch.bool)
    editable = torch.tensor([[False, True, True, True, True]])
    sample = DeletionProcess().corrupt(
        valid, editable, torch.tensor([0.5]), FixedBirths()
    )
    assert sample.edit.source.tolist() == [[0, 1, 3]]
    assert sample.gap_counts.tolist() == [[0, 0, 1, 1]]
    assert sample.gap_for_token[sample.deleted].tolist() == [2, 3]
    assert not sample.deleted[0, 0]


def test_coupled_masking_matches_linear_flex_marginals():
    torch.manual_seed(4)
    n = 30000
    valid = torch.ones(n, 1, dtype=torch.bool)
    t = torch.full((n,), 0.5)
    sample = DeletionProcess().corrupt(valid, valid, t, PowerInsertionSchedule())
    view = assembly().views[0]
    state = MaskInsertion().corrupt_training(
        view,
        torch.zeros_like(sample.edit.source),
        point(view, t),
        sample.birth_times,
        sample.edit.valid,
    )
    clean = ((state == 0) & sample.edit.valid).float().mean()
    masked = ((state == 5) & sample.edit.valid).float().mean()
    assert float(clean) == pytest.approx(
        0.5 + 0.5 * torch.log(torch.tensor(0.5)).item(), abs=0.008
    )
    assert float(masked) == pytest.approx(
        -0.5 * torch.log(torch.tensor(0.5)).item(), abs=0.008
    )


def test_forward_insertion_has_current_global_gaussian_moments():
    torch.manual_seed(17)
    view = assembly(continuous=True).views[0]
    posterior = torch.tensor([0.2, 0.8, 0, 0, 0]).expand(1, 20000, 5)
    local = point(view, torch.tensor([0.4]))
    state = ForwardInsertion("mean").initialize(view, posterior, local)
    torch.testing.assert_close(state.mean(1), 0.6 * posterior[:, 0], atol=0.01, rtol=0)
    torch.testing.assert_close(
        state.var(1), torch.full((1, 5), 0.16), atol=0.006, rtol=0
    )
    assert not torch.equal(state[:, 0], state[:, 1])


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
def test_loss_backward_sampling_and_checkpoint(continuous, feedback):
    torch.manual_seed(22)
    algo = assembly(feedback, continuous=continuous)
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 99, 99]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    editable = valid.clone()
    editable[:, 0] = False
    result = algo.loss(tokens, editable, editable, valid_mask=valid, training=True)
    assert result["loss"].isfinite()
    result["loss"].backward()
    gradients = [p.grad for p in algo.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    other = assembly(feedback, continuous=continuous)
    other.load_state_dict(algo.state_dict(), strict=True)
    prompt = torch.tensor([[1, 0, 0], [2, 0, 0]])
    fixed = torch.tensor([[True, False, False], [True, False, False]])
    samples, info = algo.generate(
        InsertionSampler(steps=3, max_length=18, overflow="truncate"), prompt, fixed
    )
    assert torch.equal(samples[:, 0], prompt[:, 0])
    assert info["nfe"] == 4
    assert samples.max() < 5 and samples.min() >= 0
    assert samples.shape[1] == max(info["lengths"])
    assert algo.training


def test_all_deleted_training_and_empty_generation():
    algo = assembly()

    class DeleteAll(PowerInsertionSchedule):
        def birth_times(self, shape, device):
            return torch.zeros(shape, device=device)

    algo.schedule = DeleteAll()
    x = torch.tensor([[1, 2], [3, 4]])
    result = algo.loss(x, torch.ones_like(x, dtype=torch.bool))
    result["loss"].backward()
    assert result["loss"].isfinite()
    with torch.no_grad():
        algo.gap_head.count.weight.zero_()
        algo.gap_head.count.bias.fill_(-100)
    output, info = algo.generate(
        InsertionSampler(steps=2), x, torch.zeros_like(x, dtype=torch.bool)
    )
    assert output.shape == (2, 0)
    assert info["lengths"] == [0, 0]


def test_hybrid_inserts_continuous_state_and_aligned_masks(monkeypatch):
    algo = assembly()
    # Isolate births from content evolution; surviving states must be untouched.
    for view in algo.views:
        view.update = lambda process, rep, state, *args, **kwargs: state
    monkeypatch.setattr(
        torch, "poisson", lambda rates: torch.tensor([[0, 2, 0]], device=rates.device)
    )
    seen = []
    original = algo.predict

    def capture(states, points, time, valid, **kwargs):
        seen.append((states, points, valid))
        return original(states, points, time, valid, **kwargs)

    algo.predict = capture
    prompt = torch.tensor([[1, 0, 2]])
    fixed = torch.tensor([[True, False, True]])
    output, metadata = algo.generate(
        InsertionSampler(steps=1, epsilon=0.4), prompt, fixed
    )
    masked, continuous = seen[-1][0]
    assert masked.tolist() == [[1, 5, 5, 2]]
    assert continuous.shape == (1, 4, 5)
    torch.testing.assert_close(continuous[:, [0, 3]], seen[0][0][1])
    assert not torch.equal(continuous[:, 1], continuous[:, 2])
    assert seen[-1][1][1].coefficients.sigma.item() == pytest.approx(0.4)
    assert output[0, [0, 3]].tolist() == [1, 2]
    assert metadata["inserted_tokens"] == [2]


def test_overflow_is_explicit_and_prompt_is_never_truncated(monkeypatch):
    algo = assembly()
    monkeypatch.setattr(
        torch, "poisson", lambda rates: torch.where(rates > 0, 10, 0).long()
    )
    prompt, fixed = torch.tensor([[1, 0]]), torch.tensor([[True, False]])
    with pytest.raises(RuntimeError, match="max_length"):
        algo.generate(InsertionSampler(steps=1, max_length=3), prompt, fixed)
    output, info = algo.generate(
        InsertionSampler(steps=1, max_length=3, overflow="truncate"), prompt, fixed
    )
    assert info["truncated"] == [True] and info["lengths"] == [3]
    assert output[0, 0] == 1


@pytest.mark.parametrize(
    "experiment,views", [("insertion_continuous", 1), ("insertion_hybrid", 2)]
)
def test_hydra_training_module_round_trip(experiment, views):
    from core.training.module import TrainingModule

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                f"+experiment={experiment}",
                "model.backbone.hidden_size=16",
                "model.backbone.n_heads=2",
                "model.backbone.n_blocks=1",
                "model.backbone.cond_dim=8",
                "model.backbone.dropout=0",
                "training.ema=0",
                "output_dir=/tmp/candi_insertion_test",
                "data.cache_dir=/tmp/candi_insertion_data",
            ],
        )
    module = TrainingModule(config)
    assert len(module.algorithm.views) == views
    module.algorithm.loss(
        torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.bool)
    )["loss"].backward()
    restored = TrainingModule(module.hparams["config"])
    restored.load_state_dict(module.state_dict(), strict=True)


@pytest.mark.parametrize("kind", ["spherical", "simplex", "uniform", "masked"])
def test_other_view_processes_use_same_structural_sampler(kind):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                "+experiment=insertion_continuous",
                f"hybrid_view@model.views.content={kind}",
                "model.backbone.hidden_size=16",
                "model.backbone.n_heads=2",
                "model.backbone.n_blocks=1",
                "model.backbone.cond_dim=8",
                "model.backbone.dropout=0",
            ],
        )
    algo = instantiate(config.variable_length, algorithm=instantiate(config.algorithm))
    x = torch.randint(0, 12, (2, 4))
    algo.loss(x, torch.ones_like(x, dtype=torch.bool))["loss"].backward()
    samples, info = algo.generate(
        InsertionSampler(steps=2, max_length=16, overflow="truncate"),
        x,
        torch.zeros_like(x, dtype=torch.bool),
    )
    assert samples.shape[1] == max(info["lengths"])
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )


def test_named_policies_survive_reversed_view_order():
    inner = make_algorithm()
    inner.model.views = torch.nn.ModuleList(reversed(inner.views))
    policies = {
        "GaussianProcess": ForwardInsertion("mean"),
        "CANDIMaskingProcess": MaskInsertion(),
    }
    algo = VariableLengthDiffusion(inner, 5, policies)
    assert isinstance(algo.insertion[0], ForwardInsertion)
    assert isinstance(algo.insertion[1], MaskInsertion)


def test_block_modifier_supports_structural_training_and_keeps_other_boundaries():
    from core.modifiers.block_diffusion.training import BlockDiffusion
    from core.modifiers.token_feedback.training import TokenFeedback
    from core.samplers.hybrid import HybridSampler

    algo = assembly()
    blocked = BlockDiffusion(algo, block_size=2, max_length=3)
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1]])
    result = blocked.loss(tokens, torch.ones_like(tokens), step=1, training=True)
    assert result["loss"].isfinite()
    assert result["num_tokens"] == tokens.numel()
    result["loss"].backward()
    validation = blocked.loss(tokens, torch.ones_like(tokens))
    assert validation["num_tokens"] == tokens.numel()
    with pytest.raises(TypeError, match="alignment"):
        TokenFeedback(algo, depth=0)
    with pytest.raises(TypeError, match="structural sampler"):
        algo.generate(
            HybridSampler(),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.bool),
        )


def test_variable_blocks_isolate_future_tokens_and_cap_each_active_block():
    from copy import deepcopy
    from core.modifiers.block_diffusion import BlockDiffusion

    torch.manual_seed(81)
    base = assembly(feedback=True, continuous=True)
    clone = deepcopy(base)
    cached = BlockDiffusion(base, block_size=2, max_length=3, cache_clean_prefix=True)
    uncached = BlockDiffusion(
        clone, block_size=2, max_length=3, cache_clean_prefix=False
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 4, 1]])
    # Only score the first block to test future-token isolation. All blocks
    # still execute in the same parallel forward pass.
    loss_mask = torch.tensor([[1, 1, 0, 0]])
    torch.manual_seed(9)
    first = cached.loss(tokens, loss_mask, step=0, training=True)["loss"]
    torch.manual_seed(9)
    second = cached.loss(changed_future, loss_mask, step=0, training=True)["loss"]
    torch.testing.assert_close(first, second)

    sampler = InsertionSampler(steps=2, max_length=20, overflow="truncate")
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    torch.manual_seed(12)
    output, metadata = cached.generate(sampler, tokens, prompt)
    assert metadata["lengths"] == [sum(metadata["block_lengths"][0])]
    assert len(metadata["block_lengths"][0]) == 2
    assert max(metadata["block_lengths"][0]) <= 3
    assert output.shape[1] == metadata["lengths"][0]

    # Variable prefixes never reuse positional KV entries, so requesting the
    # cache cannot change generation semantics.
    torch.manual_seed(12)
    other, other_metadata = uncached.generate(sampler, tokens, prompt)
    assert torch.equal(output, other)
    assert metadata["block_lengths"] == other_metadata["block_lengths"]


def test_block_sequence_features_ignore_active_padding_keys():
    from core.modifiers.block_diffusion import BlockDiffusion

    algo = BlockDiffusion(assembly(continuous=True), block_size=3, max_length=4).eval()
    time = torch.tensor([0.5])
    points = algo.algorithm.points(time)
    tokens = torch.tensor([[1, 2]])
    states = tuple(
        view.representation.clean_state(tokens) for view in algo.algorithm.views
    )
    with algo.model.clean_context(tokens[:, :0], offset=0, cache=None):
        original = algo.algorithm.predict(
            states, points, time, torch.tensor([[True, True]])
        )
        padded_states = tuple(
            torch.cat([state, view.representation.clean_state(torch.tensor([[4]]))], 1)
            for view, state in zip(algo.algorithm.views, states)
        )
        padded = algo.algorithm.predict(
            padded_states, points, time, torch.tensor([[True, True, False]])
        )
    torch.testing.assert_close(original.tokens, padded.tokens[:, :2])
    torch.testing.assert_close(original.expected_counts, padded.expected_counts[:, :3])
    torch.testing.assert_close(original.gap_tokens, padded.gap_tokens[:, :3])


def test_fully_protected_variable_block_is_copied_without_insertions():
    from core.modifiers.block_diffusion import BlockDiffusion

    algo = BlockDiffusion(assembly(continuous=True), block_size=2, max_length=4)
    tokens = torch.tensor([[1, 2, 3, 4]])
    prompt = torch.tensor([[True, True, False, False]])
    output, metadata = algo.generate(
        InsertionSampler(steps=1, max_length=20, overflow="truncate"),
        tokens,
        prompt,
    )
    assert torch.equal(output[:, :2], tokens[:, :2])
    assert metadata["block_lengths"][0][0] == 2
    assert metadata["nfe_per_sample"][0] == 2


def test_padding_does_not_change_predictions():
    torch.manual_seed(10)
    algo = assembly()
    algo.eval()
    t = torch.tensor([0.5])
    points = algo.points(t)
    tokens = torch.tensor([[1, 2, 3]])
    states = tuple(view.representation.clean_state(tokens) for view in algo.views)
    original = algo.predict(
        states, points, t, torch.ones_like(tokens, dtype=torch.bool)
    )
    padded = tuple(
        torch.cat([state, view.representation.clean_state(torch.tensor([[4, 4]]))], 1)
        for view, state in zip(algo.views, states)
    )
    extended = algo.predict(
        padded, points, t, torch.tensor([[True, True, True, False, False]])
    )
    torch.testing.assert_close(original.tokens, extended.tokens[:, :3])
    torch.testing.assert_close(
        original.expected_counts, extended.expected_counts[:, :4]
    )
    torch.testing.assert_close(original.gap_tokens, extended.gap_tokens[:, :4])


def test_training_loss_ignores_inference_shaping_and_supports_bfloat16():
    algo = assembly(True)
    tokens = torch.tensor([[1, 2, 3], [2, 3, 4]])
    active = torch.ones_like(tokens, dtype=torch.bool)
    torch.manual_seed(42)
    first = algo.loss(tokens, active)["loss"]
    algo.algorithm.predictor.temperature = 0.01
    algo.algorithm.predictor.top_k = 1
    algo.gap_predictor.temperature = 0.01
    algo.gap_predictor.top_k = 1
    torch.manual_seed(42)
    second = algo.loss(tokens, active)["loss"]
    torch.testing.assert_close(first, second)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = algo.loss(tokens, active)["loss"]
    assert result.isfinite()
    result.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in algo.parameters()
    )


def test_variable_evaluation_passes_lengths_and_reports_capacity_truncation(
    monkeypatch,
):
    from contextlib import nullcontext
    from dataclasses import dataclass
    from types import SimpleNamespace
    from core.evaluation.generation import evaluate

    @dataclass
    class Aggregate:
        records: list

    class Metrics:
        supports_variable_length = True

        def __init__(self):
            self.records = []

        def update(self, samples, examples, indices, *, lengths):
            self.records.extend(
                {"tokens": sample[:n].tolist()} for sample, n in zip(samples, lengths)
            )

        def compute(self):
            return Aggregate(self.records)

    metrics = Metrics()
    task = SimpleNamespace(
        primary_metric=SimpleNamespace(dataset="validation"),
        evaluation_dataset=lambda _: [{}, {}],
        create_metric_accumulator=lambda _: metrics,
        evaluation_batches=lambda *args: [[0, 1]],
        prepare_generation=lambda _: (
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.bool),
        ),
    )
    algo = SimpleNamespace(
        supports_variable_length=True,
        generate=lambda *args: (
            torch.tensor([[1, 2, 3], [4, 99, 99]]),
            {
                "lengths": [3, 1],
                "block_lengths": [[1, 2], [0, 1]],
                "truncated": [True, False],
                "nfe_per_sample": [2, 2],
            },
        ),
    )
    module = SimpleNamespace(
        algorithm=algo,
        device=torch.device("cpu"),
        evaluation_weights=lambda _: nullcontext(),
    )
    result = evaluate(
        module, task, {"_target_": "core.samplers.insertion.InsertionSampler"}
    )
    assert result["records"][1]["tokens"] == [4]
    assert result["generated_lengths"] == [3, 1]
    assert result["insertion_truncated"] == [True, False]
    assert result["block_lengths"] == [[1, 2], [0, 1]]
    metrics.supports_variable_length = False
    with pytest.raises(TypeError, match="task metrics"):
        evaluate(module, task, {"_target_": "core.samplers.insertion.InsertionSampler"})


def test_text8_variable_metrics_exclude_storage_padding():
    from core.data.text8 import Text8Tokenizer
    from core.tasks.text8_metrics import Text8Metrics

    tokenizer = Text8Tokenizer()
    metrics = Text8Metrics(tokenizer, {"a", "b"}, [1], "validation", "test")
    sample = torch.tensor([tokenizer.encode(" a "), tokenizer.encode(" b ")])
    examples = [{"input_ids": tokenizer.encode(" a b ")}] * 2
    metrics.update(sample, examples, [0, 1], lengths=[3, 0])
    result = metrics.compute()
    assert result.records[1].generated.character_count == 0
    assert result.records[1].generated.special_token_count == 0


@pytest.mark.parametrize("continuous", [False, True])
def test_adaptive_time_aux_feedback_includes_fully_deleted_examples(continuous):
    from core.time_sampling.loss_derivative import LossDerivativeTimeSampler

    class FixedTimes(LossDerivativeTimeSampler):
        def sample(self, count, device, **context):
            return torch.full((count,), 0.5, device=device)

    class FixedBirths(PowerInsertionSchedule):
        def birth_times(self, shape, device):
            return torch.tensor(
                [[0.1, 0.1, 0.1, 0.1], [0.8, 0.1, 0.8, 0.1], [0.8] * 4],
                device=device,
            )

    algo = assembly(continuous=continuous)
    algo.schedule = FixedBirths()
    algo.objective.count_weight = 0.3
    algo.objective.content_weight = 1.7
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]])
    mask = torch.tensor([[1.0] * 4, [1.0, 0.5, 1.0, 0.5], [0.0] * 4])
    losses, gradients = [], []
    for include in (False, True):
        algo.time_sampling_include_aux = include
        sampler = FixedTimes(adaptive=True, warmup=100, buffer_size=16)
        algo.algorithm.time_sampler = sampler
        algo.zero_grad(set_to_none=True)
        torch.manual_seed(81)
        result = algo.loss(tokens, mask, training=True)
        result["loss"].backward()
        losses.append(result["loss"].detach())
        gradients.append(
            {
                n: p.grad.clone()
                for n, p in algo.named_parameters()
                if p.grad is not None
            }
        )
        # Token-only feedback skips the all-deleted row; total feedback retains it.
        assert sampler.count.item() == (2 if include else 1)
        assert torch.isfinite(sampler.losses[: sampler.count]).all()
        if include:
            torch.testing.assert_close(
                (sampler.losses[:2] * mask.sum(-1)[:2]).sum(),
                result["loss_sum"],
            )
            assert not sampler.losses.requires_grad
        observed = sampler.count.clone()
        algo.loss(tokens, mask, training=False)
        assert sampler.count == observed
    torch.testing.assert_close(*losses)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name])


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
def test_parallel_blocks_score_every_target_in_one_fused_pass(continuous, feedback):
    from unittest.mock import patch
    from core.modifiers.block_diffusion import BlockDiffusion
    from core.time_sampling.loss_derivative import LossDerivativeTimeSampler

    algo = BlockDiffusion(
        assembly(feedback=feedback, continuous=continuous), block_size=2
    )
    algo.algorithm.algorithm.prediction.probability = 1.0
    algo.algorithm.algorithm.time_sampler = LossDerivativeTimeSampler(
        adaptive=True, warmup=0, refit_every=1, bins=2
    )
    algo.algorithm.time_sampling_include_aux = True
    tokens = torch.tensor([[1, 2, 3, 4, 1], [4, 3, 2, 1, 4]])
    weights = torch.tensor([[1.0, 0.5, 1.0, 1.0, 0.5], [1.0, 1.0, 0.5, 1.0, 1.0]])
    backbone = algo.model.model
    with patch.object(
        backbone, "forward_hidden", wraps=backbone.forward_hidden
    ) as forward:
        torch.manual_seed(712)
        result = algo.loss(
            tokens,
            weights,
            training=True,
            step=7,
            global_batch_size=2,
            world_size=1,
            rank=0,
            accumulation_steps=1,
            accumulation_index=0,
        )
    assert forward.call_count == (2 if feedback else 1)
    assert all(call.args[0].shape[0] == 2 for call in forward.call_args_list)
    assert result["num_tokens"] == weights.sum()
    torch.testing.assert_close(
        result["loss_sum"], result["loss"].detach() * weights.sum()
    )
    result["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    sampler = algo.algorithm.algorithm.time_sampler
    assert (
        sampler.count == 6
    )  # two examples, three blocks, including the short last block
    torch.testing.assert_close(
        sampler.times[:6].reshape(2, 3),
        sampler.times[:6].reshape(2, 3)[:, :1].expand(2, 3),
    )


@pytest.mark.parametrize("continuous", [False, True])
def test_parallel_block_features_and_gradients_match_isolated_contexts(continuous):
    from copy import deepcopy
    from core.modifiers.block_diffusion import BlockDiffusion
    from core.modifiers.variable_length.model import forward_features

    torch.manual_seed(401)
    base = assembly(continuous=continuous)
    copy = deepcopy(base)
    fused = BlockDiffusion(base, block_size=2).eval()
    reference = BlockDiffusion(copy, block_size=2).eval()
    clean = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    compact = torch.tensor([[1, 2], [3, 0], [4, 0], [0, 0]])
    valid = torch.tensor([[True, True], [True, False], [True, False], [False, False]])
    times = torch.tensor([0.3, 0.3, 0.7, 0.7])
    points = fused.algorithm.points(times)
    states = tuple(
        view.process.corrupt_state(
            view.representation.encode(compact), point.coefficients, valid
        )
        for view, point in zip(fused.views, points)
    )
    with fused.model.parallel_clean_context(clean, 2):
        output, counts, gaps = forward_features(
            fused.model, fused.algorithm.gap_head, states, points, times, valid
        )
    expected_logits, expected_counts, expected_gaps = [], [], []
    for index in range(4):
        start = (index % 2) * 2
        with reference.model.clean_context(
            clean[index // 2 : index // 2 + 1, :start], offset=start
        ):
            value, count, gap = forward_features(
                reference.model,
                reference.algorithm.gap_head,
                tuple(state[index : index + 1] for state in states),
                reference.algorithm.points(times[index : index + 1]),
                times[index : index + 1],
                valid[index : index + 1],
            )
        expected_logits.append(value.logits)
        expected_counts.append(count)
        expected_gaps.append(gap)
    expected = [
        torch.cat(parts) for parts in (expected_logits, expected_counts, expected_gaps)
    ]
    actual = [output.logits, counts, gaps]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in expected).backward()
    ref_params = dict(reference.named_parameters())
    for name, parameter in fused.named_parameters():
        other = ref_params[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-4, rtol=2e-4)
