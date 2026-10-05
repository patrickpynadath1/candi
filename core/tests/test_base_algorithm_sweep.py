"""The sweep catalogue must compose real compatible configurations."""

import importlib.util
from pathlib import Path
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from core.training.module import TrainingModule

spec = importlib.util.spec_from_file_location(
    "base_algorithm_sweep",
    Path(__file__).resolve().parents[2] / "examples/base_algorithm_sweep.py",
)
sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)


@pytest.fixture(scope="module")
def plan():
    args = sweep.parser().parse_args(["--seeds", "1"])
    return sweep.build_jobs(args)


def test_catalogue_counts_and_explicit_unsupported(plan):
    jobs, blocked = plan
    assert sweep.parser().parse_args([]).seeds == [1]
    assert len(jobs) == 68
    assert len({job.name for job in jobs}) == 68
    assert sum(len(job.inference) for job in jobs) == 372
    assert not blocked
    assert {variant.family for variant in sweep.method_variants()} == set(
        sweep.FAMILIES
    )
    for job in jobs:
        cfg = OmegaConf.create(job.config)
        resolved = OmegaConf.to_yaml(cfg, resolve=True)
        assert "RescaledGaussianSchedule" not in resolved
        assert "SFLMAdaptiveSchedule" not in resolved
        assert "SFLMGaussianSchedule" not in resolved
        if "OneHotRankSchedule" in resolved:
            schedule = cfg.noise_schedule.get("base", cfg.noise_schedule)
            assert schedule.inversion == "analytic"
        if job.name.startswith("simplex-"):
            assert cfg.noise_schedule.kind == "constant_linear"
            assert (
                cfg.noise_schedule.nu0,
                cfg.noise_schedule.nu1,
                cfg.noise_schedule.pivot,
            ) == (0.4, 0.75, 0.8)
        if job.name.startswith("blurring-"):
            assert cfg.noise_schedule.feature_rho == 0.5
        if job.name.startswith("gaussian-embedding"):
            assert "OneHotRankSchedule" not in resolved
            assert "OneHotArgmaxErrorSchedule" not in resolved
        if job.name.startswith("replaid"):
            assert cfg.self_conditioning.source == "posterior_mean"
            assert cfg.self_conditioning.probability == 0.25
            assert cfg.sampler.self_conditioning_mode == "bootstrap"
            assert all(
                job.sampler is None and job.overrides["evaluation.use_ema"] is False
                for job in job.inference
            )


@pytest.mark.parametrize("family", sweep.FAMILIES)
def test_all_supported_algorithm_and_sampler_assemblies(plan, family):
    jobs, _ = plan
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for job in jobs:
            if not job.name.startswith(family + "-"):
                continue
            cfg = OmegaConf.create(job.config)
            cfg.model.hidden_size = 16
            cfg.model.n_heads = 2
            cfg.model.n_blocks = 1
            cfg.model.dropout = 0.0
            if "cond_dim" in cfg.model:
                cfg.model.cond_dim = 8
            if cfg.representation._target_.endswith("EmbeddingRepresentation"):
                cfg.representation.dimension = 8
            module = TrainingModule(cfg)
            assert module.algorithm is not None
            for inference in job.inference:
                sampler_cfg = (
                    sweep.compose(overrides={"sampler": inference.sampler}).sampler
                    if inference.sampler
                    else cfg.sampler
                )
                candidate = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                candidate.sampler = OmegaConf.to_container(sampler_cfg, resolve=False)
                for key, value in inference.overrides.items():
                    OmegaConf.update(candidate, key, value)
                assert instantiate(candidate.sampler) is not None
    finally:
        torch.set_num_threads(threads)


def test_preview_never_initializes_ray(monkeypatch, capsys):
    import ray

    monkeypatch.setattr(
        ray, "init", lambda *args, **kwargs: pytest.fail("Preview launched Ray")
    )
    sweep.main(["--families", "replaid", "--seeds", "1"])
    assert "Training jobs: 2 supported" in capsys.readouterr().out


def test_uniform_plugin_self_conditioning_is_included():
    args = sweep.parser().parse_args(
        ["--families", "uniform_plugin", "--self-conditioning", "on"]
    )
    jobs, blocked = sweep.build_jobs(args)
    assert len(jobs) == 2
    assert not blocked
    assert all(job.config["self_conditioning"]["enabled"] for job in jobs)
