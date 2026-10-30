"""Seeded behavior snapshots from the samplers before their shared-loop refactor."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

FIXTURE = Path(__file__).with_name("fixtures") / "sampler_behavior.json"


def build_algorithm(family, discrete, sc):
    torch.manual_seed(123)
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=corruption_factorial",
                f"hybrid_view@model.views.content={family}",
                "model.backbone.hidden_size=16",
                "model.backbone.n_heads=2",
                "model.backbone.n_blocks=1",
                "model.backbone.cond_dim=8",
                "model.backbone.dropout=0",
                f"self_conditioning.enabled={str(sc != 'none').lower()}",
                f"self_conditioning.source={sc if sc != 'none' else 'hidden'}",
            ],
        )
    if discrete:
        OmegaConf.update(
            cfg,
            "model.views.discrete",
            OmegaConf.load(f"configs/hybrid_view/{discrete}.yaml"),
            force_add=True,
        )
    if family == "simplex":
        # These historical snapshots used FP64 states. Production presets now
        # default to FP32, which has separate precision/gradient coverage.
        cfg.model.views.content.process.precision = "float64"
    return instantiate(cfg.algorithm).eval()


def sampler_for(name):
    from core.samplers.hybrid import HybridSampler
    from core.samplers.forward_resampling import ForwardResamplingSampler
    from core.samplers.parallel_entropy_bounded import ParallelEntropyBoundedSampler

    if name == "native":
        return HybridSampler(
            steps=3, final_denoise="sample", final_transition_noise=True
        )
    if name == "native_greedy":
        return HybridSampler(steps=3, greedy_clean_estimate=True)
    if name == "forward":
        return ForwardResamplingSampler(steps=3, final_denoise="sample")
    if name == "forward_greedy":
        return ForwardResamplingSampler(steps=3, greedy_clean_estimate=True)
    return ParallelEntropyBoundedSampler(
        steps=3, freeze=name != "eb_revise", greedy_clean_estimate=name != "eb_sample"
    )


def behavior(case):
    algo = build_algorithm(case["family"], case["discrete"], case["sc"])
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    mask = torch.tensor([[True] * 5, [True, False, False, False, False]])
    traces = []

    def capture(module, args):
        states = args[0]
        h = hashlib.sha256()
        for state in states:
            h.update(state.detach().cpu().contiguous().numpy().tobytes())
        traces.append(h.hexdigest())

    with algo.model.register_forward_pre_hook(capture):
        torch.manual_seed(17)
        tokens, info = algo.generate(sampler_for(case["sampler"]), tokens, mask)
    return dict(
        tokens=tokens.tolist(),
        info=info,
        states=traces,
        rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    )


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE.read_text()) if FIXTURE.exists() else [],
    ids=lambda c: c["name"],
)
def test_legacy_behavior(case):
    assert behavior(case) == case["expected"]


if __name__ == "__main__":
    cases = []
    recipes = [
        (f, None)
        for f in [
            "gaussian",
            "simplex",
            "spherical",
            "blurring",
            "masked",
            "uniform",
            "gidd",
        ]
    ]
    recipes += [
        ("gaussian", "masked"),
        ("simplex", "uniform"),
        ("spherical", "uniform"),
    ]
    for family, discrete in recipes:
        for sc in ["none", "hidden", "posterior_mean"]:
            names = [
                "native",
                "native_greedy",
                "forward",
                "forward_greedy",
                "eb",
                "eb_sample",
            ]
            if family != "masked" and discrete != "masked":
                names += ["eb_revise"]
            for sampler in names:
                case = dict(
                    family=family,
                    discrete=discrete,
                    sc=sc,
                    sampler=sampler,
                    name=f"{family}-{discrete}-{sc}-{sampler}",
                )
                case["expected"] = behavior(case)
                cases.append(case)
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(cases, indent=2) + "\n")
    print("Captured", len(cases), "pre-refactor behaviors")
