"""Count provenance, leakage, metrics and the Sudoku-matched single-cell sweep."""

import json

import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy import sparse
import torch

from core.data.single_cell.prepare import prepare
from core.tasks.single_cell import SingleCellTask
from core.tasks.single_cell_metrics import distribution_metrics


@pytest.fixture
def prepared(tmp_path):
    ad = pytest.importorskip("anndata")
    train = ad.AnnData(np.zeros((30, 4), dtype=np.float32))
    train.var_names = ["a", "b", "c", "d"]
    train.obs_names = [f"train-{i}" for i in range(30)]
    train.obs["cell_type"] = ["B", "T"] * 15
    train.layers["X_counts"] = sparse.csr_matrix(np.tile([0, 1, 2, 4], (30, 1)))
    test = train[:6, [3, 1, 0, 2]].copy()
    test.obs_names = [f"test-{i}" for i in range(6)]
    test.layers["X_counts"] = sparse.csr_matrix(np.tile([100, 1, 0, 2], (6, 1)))
    train_path, test_path = tmp_path / "train.h5ad", tmp_path / "test.h5ad"
    train.write_h5ad(train_path)
    test.write_h5ad(test_path)
    directory = tmp_path / "prepared" / "pbmc3k"
    meta = prepare("pbmc3k", train_path, test_path, directory)
    return directory, meta, train_path, test_path


def task(prepared, conditional):
    directory, meta, *_ = prepared
    return SingleCellTask(
        {"dataset": "pbmc3k", "cache_dir": str(directory.parent)},
        meta["vocab_size"],
        meta["num_genes"] + int(conditional),
        conditional,
    )


def test_preparation_preserves_gene_order_and_fits_training_only(prepared):
    directory, meta, *_ = prepared
    assert meta["count_cap"] == 5  # test's 100 does not determine the vocabulary
    assert meta["vocab_size"] == 6  # includes zero and an overflow token
    assert json.loads((directory / "genes.json").read_text()) == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(np.load(directory / "test.npy")[0], [0, 1, 2, 5])
    training = set(np.load(directory / "train_rows.npy"))
    validation = set(np.load(directory / "validation_rows.npy"))
    assert training.isdisjoint(validation)
    assert training | validation == set(range(30))
    assert meta["splits"]["test"]["tail_fraction"] == 0.25


@pytest.mark.parametrize("conditional", [False, True])
def test_task_masks_no_expression_leakage_and_metrics(prepared, conditional):
    instance = task(prepared, conditional)
    instance.prepare_data()
    examples = [instance.validation_dataset()[i] for i in range(2)]
    batch = instance.collate_training(examples)
    assert batch["loss_mask"].sum() == 8
    assert torch.equal(batch["loss_mask"], batch["corruption_mask"])
    prompt, fixed = instance.prepare_generation(examples)
    assert fixed.sum() == 2 * conditional
    assert prompt[~fixed].eq(0).all()
    changed = [{**e, "input_ids": torch.full_like(e["input_ids"], 4)} for e in examples]
    other, other_mask = instance.prepare_generation(changed)
    assert torch.equal(prompt, other) and torch.equal(fixed, other_mask)
    metrics = instance.create_metric_accumulator("validation")
    metrics.update(batch["input_ids"], examples, [0, 1])
    result = metrics.compute()
    assert result.mmd == pytest.approx(0)
    assert result.wasserstein2 == pytest.approx(0)
    assert bool(result.per_cell_type) == conditional
    invalid = batch["input_ids"].clone()
    invalid[:, -1] = instance.vocab_size
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        metrics.update(invalid, examples, [0, 1])


def test_metrics_measure_distributions_not_paired_reconstruction():
    real = np.array([[0, 1, 3], [1, 0, 4], [2, 2, 0]])
    matched = distribution_metrics(real, real[::-1])
    assert matched["mmd"] == pytest.approx(0, abs=1e-12)
    assert matched["wasserstein2"] == pytest.approx(0)
    shifted = distribution_metrics(real, np.ones_like(real))
    assert shifted["mmd"] > 0 and shifted["wasserstein2"] > 0


def test_preparation_rejects_normalized_counts_and_overlap(prepared, tmp_path):
    import anndata as ad

    _, _, train_path, test_path = prepared
    data = ad.read_h5ad(train_path)
    data.layers["X_counts"] = sparse.csr_matrix(np.full(data.shape, 0.5))
    data.write_h5ad(tmp_path / "fractional.h5ad")
    with pytest.raises(ValueError, match="integer counts"):
        prepare("pbmc3k", tmp_path / "fractional.h5ad", test_path, tmp_path / "bad")
    with pytest.raises(ValueError, match="paths must differ"):
        prepare("pbmc3k", train_path, train_path, tmp_path / "overlap")


def test_sweep_matches_recorded_non_token_rows_and_plain_sedd(prepared):
    import csv
    from pathlib import Path
    from examples.single_cell_sweep import parser, build_jobs

    args = parser().parse_args(
        [
            "--datasets",
            "pbmc3k",
            "--modes",
            "conditional",
            "--data-cache",
            str(prepared[0].parent),
        ]
    )
    jobs = build_jobs(args, require_data=True)
    expected = {
        row["model"]
        for row in csv.DictReader(
            (
                Path(__file__).parents[2]
                / "results/sudoku-hard-5k/full-hints/summary.csv"
            ).open()
        )
        if "token" not in row["model"]
    }
    names = {
        j.name.removeprefix("pbmc3k-conditional-").removesuffix("-seed-1") for j in jobs
    }
    assert names == expected | {"sedd-none"}
    assert len(jobs) == 22
    assert sum(len(j.inference) for j in jobs) == 112
    for job in jobs:
        cfg = OmegaConf.create(job.config)
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        assert not cfg.token_feedback.enabled
        assert cfg.trainer.max_steps == 5000
        assert cfg.task.length == 5
        if "sedd-none" in job.name:
            assert not cfg.self_conditioning.enabled
            assert cfg.sampler._target_.endswith("AnalyticSampler")
            assert len(job.inference) == 2
            assert all(inference.sampler is None for inference in job.inference)
        elif cfg.self_conditioning.enabled:
            assert cfg.self_conditioning.source == "hidden"


def test_loss_backward_and_generation_for_every_family(prepared):
    from hydra.utils import instantiate
    from core.training.module import TrainingModule
    from examples.single_cell_sweep import parser, build_jobs

    args = parser().parse_args(
        [
            "--datasets",
            "pbmc3k",
            "--modes",
            "conditional",
            "--data-cache",
            str(prepared[0].parent),
        ]
    )
    instance = task(prepared, True)
    examples = [instance.training_dataset()[i] for i in range(2)]
    batch = instance.collate_training(examples)
    prompt, fixed = instance.prepare_generation(examples)
    for job in build_jobs(args, require_data=True):
        cfg = OmegaConf.create(job.config)
        model = cfg.model.get("backbone", cfg.model)
        model.hidden_size, model.n_heads, model.n_blocks, model.cond_dim = 16, 2, 1, 8
        cfg.training.ema = 0
        # Avoid the expensive production Gaussian schedule quadrature in a smoke test.
        if "views" in cfg.model and "gaussian" in job.name:
            base = cfg.model.views.first.schedule.base
            base.grid_size, base.quadrature_order = 33, 8
        module = TrainingModule(cfg)
        loss = module.algorithm.loss(
            batch["input_ids"],
            batch["loss_mask"],
            corruption_mask=batch["corruption_mask"],
        )["loss"]
        assert torch.isfinite(loss), job.name
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in module.parameters()
            if p.grad is not None
        )
        module.algorithm.eval()
        # Exercise each distinct sampler/policy once, with small transition budgets.
        seen = set()
        for inference in job.inference:
            key = (inference.sampler, inference.overrides.get("sampler.freeze"))
            if key in seen:
                continue
            seen.add(key)
            sampler_cfg = (
                instantiate_sampler_config(inference.sampler)
                if inference.sampler
                else cfg.sampler.copy()
            )
            for field, value in inference.overrides.items():
                if field.startswith("sampler."):
                    OmegaConf.update(sampler_cfg, field[8:], value, force_add=True)
            for field in ("steps", "max_steps"):
                if field in sampler_cfg:
                    sampler_cfg[field] = 2
            with torch.no_grad():
                generated, _ = module.algorithm.generate(
                    instantiate(sampler_cfg), prompt, fixed
                )
            assert generated.shape == prompt.shape, job.name
            assert torch.equal(generated[fixed], prompt[fixed]), job.name
            assert (
                (generated[:, 1:] >= 0) & (generated[:, 1:] < instance.vocab_size)
            ).all(), job.name


def instantiate_sampler_config(name):
    from core.sweeps import compose

    return compose(overrides={"sampler": name}).sampler


def test_training_checkpoint_evaluation_lifecycle(prepared, tmp_path):
    from core.sweeps import compose
    from core.training.runner import run

    directory, metadata, *_ = prepared
    cfg = compose(
        "smoke",
        {
            "task": "single_cell",
            "data": "pbmc3k",
            "data.cache_dir": str(directory.parent),
            "data.vocab_size": metadata["vocab_size"],
            "data.num_genes": metadata["num_genes"],
            "task.length": metadata["num_genes"] + 1,
            "task.conditional": True,
            "output_dir": str(tmp_path / "run"),
        },
    )
    cfg.loader.global_batch_size = 2
    cfg.loader.eval_batch_size = 2
    cfg.loader.num_workers = 0
    cfg.trainer.max_steps = 1
    cfg.trainer.val_check_interval = 1
    cfg.trainer.accumulate_grad_batches = 1
    cfg.trainer.enable_progress_bar = False
    cfg.checkpoint.every_n_steps = 1
    cfg.model.hidden_size, cfg.model.n_heads, cfg.model.n_blocks = 16, 2, 1
    cfg.callback.every_n_steps = 1
    cfg.callback.subset_size = 2
    run(cfg)
    output = tmp_path / "run"
    assert (output / "checkpoints/last.ckpt").is_file()
    assert (output / "task_callback/step_00000001.json").is_file()
    cfg.mode = "evaluate"
    cfg.evaluation.checkpoint = str(output / "checkpoints/last.ckpt")
    cfg.evaluation.device = "cpu"
    cfg.evaluation.max_examples = 2
    run(cfg)
    result = json.loads((output / "results.json").read_text())
    assert result["num_total"] == 2
    assert result["mean_nfe"] > 0
    assert len(result["records"]) == 2
    assert result["records"][0]["ground_truth"][-1] == metadata["count_cap"]
