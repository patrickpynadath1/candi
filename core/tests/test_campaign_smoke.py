import json
import shlex
from types import SimpleNamespace

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from core.cluster.slurm import entrypoint, node_command, prepare_ray_worker
from core.sweeps import smoke, nested_tuning
from core.tests.test_programmatic_sweeps import tiny
from core.sweeps.configuration import compose


def case(tmp_path):
    cfg = compose(
        "corruption_factorial",
        [
            "hybrid_view@model.views.content=masked",
            "model.backbone.hidden_size=16",
            "model.backbone.cond_dim=8",
            "model.backbone.n_heads=2",
            "model.backbone.n_blocks=1",
        ],
    )
    small = OmegaConf.create(tiny())
    cfg.task = small.task
    cfg.trainer = small.trainer
    cfg.loader = small.loader
    path = tmp_path / "base.yaml"
    OmegaConf.save(cfg, path)
    return {
        "name": "tiny",
        "config": str(path),
        "resources": {"cpu": 1, "gpu": 0},
        "training": {"space": {}, "trials": 1},
        "evaluation": {
            "tuning_indices": [0],
            "test_indices": [1],
            "seeds": [41],
            "test_seeds": [1041],
            "nfe": 128,
            "direction": "maximize",
            "score": {"kind": "metric", "metric": "accuracy"},
        },
        "inference": [
            {
                "name": name,
                "sampler": {"_target_": target, "steps": 127},
                "budget_parameter": "steps",
                "space": {},
                "trials": 1,
            }
            for name, target in [
                ("ancestral", "core.samplers.hybrid.HybridSampler"),
                (
                    "renoising",
                    "core.samplers.forward_resampling.ForwardResamplingSampler",
                ),
            ]
        ],
    }


def test_real_training_checkpoint_and_each_sampler(tmp_path, monkeypatch):
    spec = case(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([spec]))
    source = (tmp_path / "base.yaml").read_bytes()
    calls = []

    def direct(stage, payload, output):
        assert "test_indices" not in payload.get(
            "evaluation", payload.get("validation", {})
        )
        calls.append(stage)
        worker = (
            nested_tuning._train_worker
            if stage == "train"
            else nested_tuning._evaluate_worker
        )
        return worker(payload, output)

    monkeypatch.setattr(smoke, "execute", direct)
    report = smoke.run(manifest, tmp_path / "smoke", no_compile=True)
    assert report["passed"], report
    assert calls == ["train", "evaluate", "evaluate"]
    checkpoint = torch.load(
        tmp_path / "smoke/tiny/training/checkpoints/last.ckpt",
        weights_only=False,
        map_location="cpu",
    )
    assert checkpoint["global_step"] == 1
    assert (tmp_path / "base.yaml").read_bytes() == source
    for policy in spec["inference"]:
        result = json.loads(
            (
                tmp_path
                / "smoke/tiny/evaluation"
                / policy["name"]
                / "seed-41/results.json"
            ).read_text()
        )
        assert result["max_nfe"] <= 2
    with pytest.raises(FileExistsError):
        smoke.run(manifest, tmp_path / "smoke")


def test_failed_policy_does_not_skip_other_policies(tmp_path, monkeypatch):
    spec = case(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([spec]))

    def execute(stage, payload, output):
        if stage == "train":
            return {"config": spec["config"], "checkpoint": "dummy"}
        if payload["policy"]["name"] == "ancestral":
            raise RuntimeError("deliberate failure")
        return {"score": 0.0}

    monkeypatch.setattr(smoke, "execute", execute)
    report = smoke.run(manifest, tmp_path / "smoke")
    assert not report["passed"]
    assert [x["status"] for x in report["cases"][0]["policies"]] == ["failed", "passed"]


def test_block_budget_and_ar_cap(tmp_path):
    spec = case(tmp_path)
    cfg = OmegaConf.load(spec["config"])
    cfg.block_diffusion = {"enabled": True, "block_size": 2}
    policy, budget = smoke.policy_budget(cfg, spec["inference"][1], {}, 1)
    assert budget == 8  # Four blocks, each one transition plus readout.
    ar = {
        "name": "ar",
        "sampler": {"_target_": "core.samplers.autoregressive.AutoregressiveSampler"},
        "budget_parameter": None,
    }
    selected, nfe = smoke.policy_budget(cfg, ar, {}, 1)
    assert nfe == 1 and selected["sampler"]["max_new_tokens"] == 1
    assert "max_new_tokens" not in ar["sampler"]


def test_ar_one_token_preserves_prompt_and_pads_remainder():
    from core.tests.test_ar_d3pm import config
    from core.samplers.autoregressive import AutoregressiveSampler

    algo = instantiate(
        config("autoregressive", "parameterization.commitment=greedy").algorithm
    )
    tokens = torch.randint(1, 9, (2, 10))
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :5] = True
    result, stats = algo.generate(AutoregressiveSampler(max_new_tokens=1), tokens, mask)
    assert stats["nfe_per_sample"] == [1, 1]
    assert torch.equal(result[mask], tokens[mask])
    assert (result[0, 4:] == 0).all() and (result[1, 6:] == 0).all()
    with pytest.raises(ValueError):
        AutoregressiveSampler(max_new_tokens=0)


def test_cluster_commands_and_quoted_entrypoint(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = SimpleNamespace(cpus=8, gpus=2, port=6380, dashboard_port=8266)
    head = node_command(args, True, "10.0.0.1")
    worker = node_command(args, False, "10.0.0.1")
    assert "--head" in head and "--block" in head
    assert "--head" not in worker and "10.0.0.1:6380" in worker
    args = SimpleNamespace(
        smoke=True,
        manifest=tmp_path / "manifest with spaces.json",
        output=tmp_path / "smoke",
        train_steps=1,
        sampler_steps=1,
        examples=1,
        no_compile=True,
    )
    info = {
        "python": "/shared/venv/bin/python",
        "source": "/shared/source with spaces",
        "address": "10.0.0.1:6380",
    }
    words = shlex.split(entrypoint(info, args))
    assert words[:4] == ["cd", info["source"], "&&", "exec"]
    assert str(args.manifest) in words and "core.sweeps.smoke" in words


def test_ray_worker_does_not_inherit_slurm_distributed_ranks(monkeypatch):
    import os
    from lightning.fabric.plugins.environments import SLURMEnvironment

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_NTASKS", "30")
    monkeypatch.setenv("SLURM_NNODES", "30")
    monkeypatch.setenv("SLURM_NODEID", "8")
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("RAY_AUTH_TOKEN_PATH", "/shared/private-token")
    prepare_ray_worker()
    assert not SLURMEnvironment.detect()
    assert "RANK" not in os.environ
    assert os.environ["CANDI_SLURM_JOB_ID"] == "123"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["RAY_AUTH_TOKEN_PATH"] == "/shared/private-token"
