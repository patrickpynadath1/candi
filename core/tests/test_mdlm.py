"""Mathematical and orchestration checks independent of the legacy repository."""

from omegaconf import OmegaConf
from pathlib import Path
import ast
import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir

from core.algorithms.mdlm import MDLM
from hydra.utils import instantiate
from core.objectives.weighted_ce import WeightedCrossEntropy
from core.parameterizations.posterior import PosteriorParameterization
from core.processes.absorbing import AbsorbingProcess
from core.noise_schedules.log_linear import LogLinear
from core.time_sampling.uniform import UniformTimeSampler
from core.samplers.ancestral import AncestralSampler
from core.samplers.entropy_bounded import EntropyBoundedSampler, select_prefix
from core.training.module import TrainingModule
from core.training.checkpoints import load_weights
from core.data.sudoku import SudokuTokenizer, encode_example, sudoku_metrics


def config(*overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=smoke",
                "output_dir=/tmp/core_test",
                "data.cache_dir=/tmp/core_test_data",
                *overrides,
            ],
        )


def test_training_and_validation_posterior_ignore_sampling_configuration():
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0, 100.0], [3.0, 1.0, 2.0, 100.0]]], requires_grad=True
    )
    state = torch.tensor([[3, 1]])
    predictor = PosteriorParameterization(3)
    output = predictor.training_posterior(logits, state)
    other = PosteriorParameterization(
        3, temperature=0.01, top_k=1, top_p=0.1
    ).training_posterior(logits, state)
    torch.testing.assert_close(output, other)
    assert output.exp()[0, 0, 3] == 0
    torch.testing.assert_close(output.exp()[0, 1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    loss = WeightedCrossEntropy()(
        output, torch.tensor([[2, 1]]), torch.tensor([0.4]), torch.tensor([-0.999])
    )
    expected = -torch.log_softmax(logits[0, 0, :3], -1)[2] * 0.999 / 0.6
    torch.testing.assert_close(loss.sum(), expected)
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].count_nonzero() == 0


@pytest.mark.parametrize(
    "shaping", [{"temperature": 0.5}, {"top_k": 1}, {"top_p": 0.5}, {}]
)
def test_shaping_preserves_support_and_visible_tokens(shaping):
    predictor = PosteriorParameterization(3, **shaping)
    logits = torch.tensor([[[0.0, 1.0, 2.0, 50.0], [5.0, 2.0, 0.0, 50.0]]])
    state = torch.tensor([[3, 2]])
    probabilities = predictor.sampling_posterior(logits, state).exp()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, 2))
    assert not probabilities[..., 3].any()
    assert probabilities[0, 1, 2] == 1


def test_forward_and_reverse_empirical_probabilities():
    torch.manual_seed(19)
    process = AbsorbingProcess(2)
    clean = torch.zeros(20000, 2, dtype=torch.long)
    active = torch.tensor([True, False]).expand_as(clean)
    corrupt = process.corrupt(clean, torch.full((len(clean),), 0.3), active)
    assert abs(float(corrupt[:, 0].eq(2).float().mean()) - 0.7) < 0.012
    assert not corrupt[:, 1].any()
    state = torch.tensor([2, 1]).expand_as(clean)
    posterior = torch.tensor([0.25, 0.75, 0.0]).expand(len(clean), 2, 3)
    result = process.reverse(
        state, posterior, torch.full((len(clean),), 0.2), torch.full((len(clean),), 0.6)
    )
    observed = torch.bincount(result[:, 0], minlength=3) / len(clean)
    torch.testing.assert_close(
        observed, torch.tensor([0.125, 0.375, 0.5]), atol=0.012, rtol=0
    )
    assert result[:, 1].eq(1).all()


def test_time_sampling_covers_global_strata_and_accumulation_partitions():
    sampler = UniformTimeSampler()
    torch.manual_seed(5)
    all_times = sampler.sample(16, "cpu")
    strata = torch.floor((all_times - 0.001) / 0.999 * 16)
    torch.testing.assert_close(strata, torch.arange(16).float())
    for rank in range(2):
        for accumulation in range(2):
            torch.manual_seed(5)
            part = sampler.sample(
                4,
                "cpu",
                global_batch_size=16,
                world_size=2,
                rank=rank,
                accumulation_steps=2,
                accumulation_index=accumulation,
            )
            torch.testing.assert_close(
                part, all_times.reshape(2, 2, 4)[rank, accumulation]
            )


def test_schedule_derivative_and_loss_endpoints():
    schedule = LogLinear()
    time = torch.tensor([0.1, 0.5, 1.0], requires_grad=True)
    keep, derivative = schedule(time)
    torch.testing.assert_close(torch.autograd.grad(keep.sum(), time)[0], derivative)
    assert abs(keep[-1].item() - 0.001) < 1e-7
    with pytest.raises(ValueError):
        WeightedCrossEntropy()(
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1),
            -torch.ones(1),
        )


class Oracle(nn.Module):
    def forward(self, tokens, conditioning):
        values = torch.full((*tokens.shape, 4), -100.0, device=tokens.device)
        values[..., 1] = 0
        return values


def oracle():
    return MDLM(
        Oracle(),
        PosteriorParameterization(3),
        AbsorbingProcess(3),
        LogLinear(),
        UniformTimeSampler(),
        WeightedCrossEntropy(),
        length=5,
    )


@pytest.mark.parametrize(
    "sampler", [AncestralSampler(4), EntropyBoundedSampler(0.1, 5)]
)
def test_generation_preserves_prompt_and_counts_model_evaluations(sampler):
    algorithm = oracle()
    prompt = torch.tensor([[2, 0, 0, 0, 0], [2, 2, 2, 2, 2]])
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    calls = []
    handle = algorithm.model.register_forward_hook(lambda *args: calls.append(1))
    samples, metadata = algorithm.generate(sampler, prompt, mask)
    handle.remove()
    assert samples.tolist() == [[2, 1, 1, 1, 1], [2, 2, 2, 2, 2]]
    assert metadata["nfe"] == len(calls)
    assert algorithm.training


def test_entropy_is_measured_before_greedy_commitment():
    class Uncertain(nn.Module):
        def forward(self, tokens, conditioning):
            return torch.zeros(*tokens.shape, 4)

    algorithm = oracle()
    algorithm.model = Uncertain()
    prompt = torch.zeros(1, 5, dtype=torch.long)
    _, metadata = algorithm.generate(EntropyBoundedSampler(0, 5), prompt, prompt.bool())
    assert metadata["nfe"] == 5
    assert metadata["accepted_per_step"] == [[1]] * 5


def test_entropy_budget_and_exhaustion():
    entropy = torch.tensor([[0.05, 0.08, 0.9]])
    assert select_prefix(
        entropy, torch.ones_like(entropy, dtype=torch.bool), 0.1
    ).tolist() == [[True, True, False]]
    sampler = EntropyBoundedSampler(0, 1)

    class Uncertain(nn.Module):
        def forward(self, tokens, conditioning):
            return torch.zeros(*tokens.shape, 4)

    algorithm = oracle()
    algorithm.model = Uncertain()
    calls = []
    hook = algorithm.model.register_forward_hook(lambda *args: calls.append(1))
    prompt = torch.zeros(1, 5, dtype=torch.long)
    with pytest.raises(RuntimeError, match="exceeded"):
        algorithm.generate(sampler, prompt, prompt.bool())
    hook.remove()
    assert len(calls) == 1


def test_real_dit_backward_and_native_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    settings = config()
    module = TrainingModule(settings)
    tokens = torch.randint(0, 12, (2, 180))
    mask = torch.zeros_like(tokens)
    mask[:, 91:] = 1
    module.algorithm.loss(tokens, mask)["loss"].backward()
    assert module.algorithm.model.output_layer.linear.weight.grad.count_nonzero() > 0
    module.ema.update(module.algorithm.parameters())
    checkpoint = {"state_dict": module.state_dict(), "global_step": 4}
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    restored = TrainingModule(settings)
    assert load_weights(restored, path) == 4
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    assert restored.ema.num_updates == module.ema.num_updates
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with restored.evaluation_weights():
        pass
    for key, value in before.items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_all_sampler_configs_construct():
    for name in ("ancestral", "entropy_bounded"):
        settings = config(f"sampler={name}")
        assert instantiate(settings.sampler)
        assert instantiate(settings.algorithm)


def test_recursive_config_construction_tracks_overrides_and_serializes():
    settings = config(
        "algorithm=mdlm",
        "task.vocab_size=17",
        "task.length=20",
        "parameterization.temperature=0.7",
        "noise_schedule.epsilon=0.02",
    )
    # A resolved checkpoint configuration must construct the same graph without
    # Hydra runtime state or a custom factory.
    settings = OmegaConf.create(OmegaConf.to_container(settings, resolve=True))
    algorithm = instantiate(settings.algorithm)
    assert algorithm.length == 20
    assert algorithm.predictor.mask_index == algorithm.process.mask_index == 17
    assert algorithm.predictor.temperature == 0.7
    keep, _ = algorithm.schedule(torch.ones(1))
    torch.testing.assert_close(keep, torch.tensor([0.02]))
    logits = algorithm.model_logits(torch.zeros(1, 20, dtype=torch.long), torch.ones(1))
    assert logits.shape == (1, 20, 18)
    # Selection isn't hardcoded to MDLM in the caller.
    settings.algorithm._target_ = "types.SimpleNamespace"
    alternative = instantiate(settings.algorithm)
    assert alternative.model.__class__ == algorithm.model.__class__
    assert alternative.predictor.mask_index == 17


def test_zero_warmup_starts_at_configured_learning_rate():
    settings = config("training.warmup_steps=0")
    optimizer = TrainingModule(settings).configure_optimizers()["optimizer"]
    assert optimizer.param_groups[0]["lr"] == settings.optimizer.lr


def test_sudoku_transport_and_metrics():
    grid = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    tokens, mask = encode_example(grid, grid, SudokuTokenizer())
    assert len(tokens) == 180 and sum(mask) == 89
    metrics = sudoku_metrics(tokens[91:], tokens[91:], tokens[1:90])
    assert metrics["exact"] and metrics["valid_solution"]


def test_core_never_imports_legacy_modules():
    forbidden = {
        "algos",
        "models",
        "samplers",
        "tasks",
        "dataloader",
        "utils",
        "metrics",
        "main",
    }
    for path in Path(__file__).parents[1].rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name.split(".")[0] in forbidden for alias in node.names
                ), path
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] not in forbidden, path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rotary_cache_remains_fp32_under_gpu_autocast():
    model = instantiate(config().algorithm).model.cuda().eval()
    tokens = torch.zeros(1, 180, dtype=torch.long, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(tokens, torch.zeros(1, device="cuda"))
    assert model.rotary_emb.cos_cached.dtype == torch.float32
    frequency = torch.einsum(
        "i,j->ij", torch.arange(180, device="cuda").float(), model.rotary_emb.inv_freq
    )
    expected = torch.cat([frequency, frequency], -1).cos()
    torch.testing.assert_close(model.rotary_emb.cos_cached[0, :, 0, 0], expected)


def test_training_resume_and_both_evaluators(tmp_path):
    import json
    from core.training.runner import run

    settings = config("callback.every_n_steps=2", "callback.subset_size=2")
    settings.output_dir = str(tmp_path / "train")
    settings.data.cache_dir = str(tmp_path / "data")
    settings.trainer.enable_progress_bar = False
    run(settings)
    assert (tmp_path / "train/task_callback/step_00000002.json").exists()
    path = tmp_path / "train/checkpoints/last.ckpt"
    first = torch.load(path, weights_only=False)
    assert first["global_step"] == 2 and first["ema"]["num_updates"] == 2
    settings.checkpoint.resume = str(path)
    settings.trainer.max_steps = 4
    settings.output_dir = str(tmp_path / "resume")
    run(settings)
    resumed_path = tmp_path / "resume/checkpoints/last.ckpt"
    resumed = torch.load(resumed_path, weights_only=False)
    assert resumed["global_step"] == 4 and resumed["ema"]["num_updates"] == 4
    assert resumed["lr_schedulers"][0]["last_epoch"] == 4
    for name in ("ancestral", "entropy_bounded"):
        evaluation = config(f"sampler={name}")
        evaluation.mode = "evaluate"
        evaluation.output_dir = str(tmp_path / name)
        evaluation.data.cache_dir = settings.data.cache_dir
        evaluation.evaluation.checkpoint = str(resumed_path)
        run(evaluation)
        results = json.loads((tmp_path / name / "results.json").read_text())
        assert results["num_total"] == 2
        assert results["checkpoint_step"] == 4
        assert all(record["nfe"] > 0 for record in results["records"])


def test_legacy_tokenizer_unpickles_without_legacy_imports(tmp_path, monkeypatch):
    import sys
    import types
    from core.training import legacy_pickle

    package = types.ModuleType("tasks")
    module = types.ModuleType("tasks.sudoku")
    cls = type("SudokuTokenizer", (), {"__module__": "tasks.sudoku"})
    module.SudokuTokenizer = cls
    package.sudoku = module
    path = tmp_path / "legacy.ckpt"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "tasks", package)
        patch.setitem(sys.modules, "tasks.sudoku", module)
        torch.save({"tokenizer": cls()}, path)
    restored = torch.load(path, weights_only=False, pickle_module=legacy_pickle)
    assert isinstance(restored["tokenizer"], SudokuTokenizer)
