"""Executable component combinations and selection without held-out leakage."""

import json

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from core.sweeps import compose
from examples import corruption_self_conditioning_sweep as sweep
from examples.report_corruption_self_conditioning import (
    report,
    select_policies,
    summarize_records,
)


@pytest.fixture(scope="module")
def catalogue():
    args = sweep.parser().parse_args(["--seeds", "1"])
    return sweep.build_jobs(args)


def inference_config(job, choice):
    cfg = OmegaConf.create(job.config)
    cfg.sampler = compose(overrides={"sampler": choice.sampler}).sampler
    for key, value in choice.overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return cfg


def test_balanced_defaults_and_protocol(catalogue):
    args = sweep.parser().parse_args([])
    assert args.seeds == [1, 2, 3, 4, 5]
    assert len(catalogue) == 12
    assert len({j.name for j in catalogue}) == 12
    for job in catalogue:
        cfg = OmegaConf.create(job.config)
        assert cfg.wandb.project == "self-cond-sweep"
        assert cfg.wandb.mode == "online"
        assert cfg.wandb.name == job.name
        assert cfg.trainer.max_steps == 10_000
        assert cfg.checkpoint.every_n_steps == 1000
        assert (
            cfg.trainer.val_check_interval == 1000 * cfg.trainer.accumulate_grad_batches
        )
        assert cfg.trainer.precision == "bf16-mixed"
        assert cfg.data.num_train == 48_000 and cfg.data.num_valid == 2_000
        assert cfg.self_conditioning.source == "hidden"
        assert cfg.model.time_conditioning == "identity"
        assert cfg.objective._target_.endswith("CrossEntropy")
        assert cfg.time_sampler._target_.endswith("UniformTimeSampler")
        assert cfg.callback.every_n_steps == 1000
        assert set(cfg.callback.sampler_configs) == set(sweep.POLICIES)
        assert not cfg.block_diffusion.enabled and not cfg.token_feedback.enabled
        assert len(cfg.model.views) == 1 and len(job.inference) == 3
        for choice in job.inference:
            assert choice.overrides["seed"] == 1729
            assert choice.overrides["sampler.steps"] == 127
        view = cfg.model.views.content
        if cfg.factorial.family == "spherical":
            assert view.schedule.auto_truncation is False
        if cfg.factorial.family == "gaussian":
            assert view.schedule._target_.endswith("CDCDGaussianSchedule")
            assert view.schedule.base.coupling == "linear"
            assert cfg.factorial.schedule_adaptation == "cdcd"
            assert view.representation.state_representation._target_.endswith(
                "OneHotRepresentation"
            )
    plan = sweep.make_plan(args, catalogue)
    protocol = plan["protocol"]
    assert len(protocol["selection_indices"]) == len(protocol["report_indices"]) == 1000
    assert not set(protocol["selection_indices"]) & set(protocol["report_indices"])
    assert sorted(protocol["selection_indices"] + protocol["report_indices"]) == list(
        range(2000)
    )
    assert sweep.make_plan(args, catalogue) == plan
    args.entropy_bound = 0.2
    changed = sweep.make_plan(args, sweep.build_jobs(args))
    assert changed["fingerprint"] != plan["fingerprint"]


@pytest.mark.parametrize("index", range(12))
def test_every_cell_backward_and_real_128_nfe_samplers(catalogue, index):
    job = catalogue[index]
    cfg = OmegaConf.create(job.config)
    cfg.model.backbone.hidden_size = 16
    cfg.model.backbone.cond_dim = 8
    cfg.model.backbone.n_blocks = 1
    cfg.model.backbone.n_heads = 2
    cfg.model.backbone.dropout = 0
    algo = instantiate(cfg.algorithm)
    assert algo.prediction.enabled == cfg.self_conditioning.enabled
    torch.manual_seed(31)
    tokens = torch.randint(0, 12, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :2] = False
    result = algo.loss(tokens, mask, training=True)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    for choice in job.inference:
        candidate = inference_config(job, choice)
        sampler = instantiate(candidate.sampler)
        calls = []
        # Observe the actual posterior passed to each native transition.
        observed = []
        original_update = algo.views[0].update

        def capture(*args, **kwargs):
            observed.append(args[3].detach().clone())
            return original_update(*args, **kwargs)

        algo.views[0].update = capture
        handle = algo.model.register_forward_hook(lambda *args: calls.append(1))
        try:
            output, metadata = algo.generate(sampler, tokens, ~mask)
        finally:
            handle.remove()
            algo.views[0].update = original_update
        assert torch.equal(output[:, :2], tokens[:, :2])
        assert output.min() >= 0 and output.max() < 12
        assert metadata["nfe"] == len(calls)
        if choice.sampler == "hybrid":
            assert len(observed) == 127
            if sampler.greedy_clean_estimate:
                assert all(torch.all((p == 0) | (p == 1)) for p in observed)
                assert all(torch.all(p.sum(-1) == 1) for p in observed)
            assert len(calls) == 128
            assert metadata["nfe_per_sample"] == [128, 128]
        else:
            assert 0 < len(calls) <= 128
            assert all(0 < n <= len(calls) for n in metadata["nfe_per_sample"])


def test_adaptive_gaussian_feedback_and_checkpoint(catalogue):
    job = next(
        job for job in catalogue if job.config["factorial"]["family"] == "gaussian"
    )
    cfg = OmegaConf.create(job.config)
    cfg.model.backbone.hidden_size = 16
    cfg.model.backbone.cond_dim = 8
    cfg.model.backbone.n_blocks = 1
    cfg.model.backbone.n_heads = 2
    schedule = cfg.model.views.content.schedule
    schedule.buffer_size = 32
    schedule.warmup_steps = 0
    schedule.refit_every = 1
    algo = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 12, (32, 6))
    algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool), step=1, training=True)
    clock = algo.views[0].schedule.clock
    assert clock.num_observed.item() == 32
    assert clock.last_refit_step.item() == 1
    assert clock.has_schedule.item()
    restored = instantiate(cfg.algorithm)
    restored.load_state_dict(algo.state_dict())
    times = torch.linspace(0.01, 0.99, 20)
    torch.testing.assert_close(clock(times), restored.views[0].schedule.clock(times))
    sampler = instantiate(inference_config(job, job.inference[0]).sampler)
    generated, metadata = restored.generate(
        sampler, tokens[:2], torch.zeros_like(tokens[:2], dtype=torch.bool)
    )
    assert generated.min() >= 0 and generated.max() < 12
    assert metadata["nfe"] == 128


def test_preview_does_not_launch_ray(monkeypatch, tmp_path, capsys):
    import ray

    monkeypatch.setattr(
        ray, "init", lambda *a, **k: pytest.fail("Preview launched Ray")
    )
    path = tmp_path / "plan.json"
    sweep.main(["--families", "masked", "--seeds", "1", "--write-plan", str(path)])
    assert "Training: 2 runs x 10,000 steps" in capsys.readouterr().out
    assert len(json.loads(path.read_text())["jobs"]) == 2


def small_plan():
    args = sweep.parser().parse_args(
        [
            "--families",
            "masked",
            "--seeds",
            "1",
            "2",
            "--eval-examples",
            "4",
        ]
    )
    jobs = sweep.build_jobs(args)
    return jobs, sweep.make_plan(args, jobs)


def test_selection_uses_selection_split_and_all_seeds():
    _, plan = small_plan()
    rows = []
    for job in plan["jobs"]:
        f = job["config"]["factorial"]
        for policy in sweep.POLICIES:
            # Per-seed maxima would pick different samplers. Mean selection
            # performance picks native for BOTH seeds, despite EB winning test.
            selection = (
                (100 if f["training_seed"] == 1 else 0)
                if policy == "native_128"
                else 40
            )
            rows.append(
                {
                    **f,
                    "policy": policy,
                    "selection_success_percent": selection,
                    "selection_mean_nfe": 128 if policy == "native_128" else 5,
                    "report_success_percent": 0 if policy == "native_128" else 100,
                }
            )
    selected, missing = select_policies(rows, plan)
    assert not missing
    assert len(selected) == 4
    assert {r["selected_policy"] for r in selected} == {"native_128"}
    selected, missing = select_policies(rows[:-1], plan)
    assert len(selected) == 2 and len(missing) == 1


def test_report_end_to_end_and_reject_changed_budget(tmp_path):
    jobs, plan = small_plan()
    (tmp_path / "job-plan.json").write_text(json.dumps(plan))
    paths = []
    selection = set(plan["protocol"]["selection_indices"])
    for index, job in enumerate(jobs):
        for policy_index, choice in enumerate(job.inference):
            output = tmp_path / f"trial-{index}-{policy_index}" / "run"
            output.mkdir(parents=True)
            cfg = inference_config(job, choice)
            cfg.output_dir = str(output)
            cfg.mode = "evaluate"
            OmegaConf.save(
                OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
                output / "config.yaml",
            )
            records = [
                {
                    "index": i,
                    "ground_truth": str(i),
                    "valid_solution": (i in selection) == (policy_index == 0),
                    "nfe": 128 if choice.sampler == "hybrid" else 5,
                }
                for i in range(4)
            ]
            result = {"records": records, "checkpoint_step": 10_000}
            path = output / "results.json"
            path.write_text(json.dumps(result))
            paths.append(path)
    result = report(tmp_path)
    assert not result["missing"] and len(result["selected"]) == 4
    assert all(r["report_success_percent"] == 0 for r in result["selected"])
    assert (tmp_path / "analysis" / "selected-per-seed.csv").exists()
    status = json.loads((tmp_path / "analysis" / "status.json").read_text())
    assert status["complete"]
    changed = json.loads(paths[0].read_text())
    changed["records"][0]["nfe"] = 129
    paths[0].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="exactly 128"):
        report(tmp_path)


def test_duplicate_puzzle_rejected():
    protocol = {"selection_indices": [0], "report_indices": [1]}
    with pytest.raises(ValueError, match="exactly once"):
        summarize_records([{"index": 0}, {"index": 0}], protocol, "native_128")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--seeds", "1", "1"],
        ["--entropy-bound", "nan"],
        ["--eval-examples", "1"],
        ["--train-steps", "0"],
        ["--resume"],
    ],
)
def test_invalid_plans_fail(arguments):
    with pytest.raises(SystemExit):
        sweep.main(arguments)
