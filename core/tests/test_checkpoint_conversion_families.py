"""Old checkpoints of every migrated family convert, load and sample the same.

An "old" checkpoint is written by a TrainingModule running the frozen standalone
class (core/tests/reference_algorithms). It must load strictly into today's
module after convert_checkpoint, with every tensor and EMA shadow equal (matched
through the converter's own name map and permutation, since ContinuousDiT
families list their parameters in another order), and sample the same tokens
and NFE as the old class under its old sampler.
"""

from hydra.utils import instantiate
import pytest
import torch

from core.models.legacy_checkpoints import conversion_plan, convert_checkpoint
from core.tests.test_checkpoint_conversion import config, perturb, saved
from core.tests.view_equivalence import batch
from core.training.module import TrainingModule

REF = "core.tests.reference_algorithms"
OLD = "core.tests.reference_samplers"


def sampler(target, **options):
    return {"_target_": target, **options}


# name: (experiment, frozen class, overrides, (current sampler, old sampler))
CASES = {
    "gaussian": (
        "gaussian_diffusion_sudoku",
        f"{REF}.gaussian_diffusion.GaussianDiffusion",
        {},
        (sampler("core.samplers.euler.EulerSampler", steps=4),
         sampler(f"{OLD}.continuous.EulerSampler", steps=4)),
    ),
    "gaussian_sc": (
        "gaussian_diffusion_sudoku",
        f"{REF}.gaussian_diffusion.GaussianDiffusion",
        {"self_conditioning.enabled": True},
        (sampler("core.samplers.euler.EulerSampler", steps=4),
         sampler(f"{OLD}.continuous.EulerSampler", steps=4)),
    ),
    "blurring": (
        "sudoku_blurring_hard",
        f"{REF}.blurring_diffusion.BlurringDiffusion",
        {},
        (sampler("core.samplers.blur_ancestral.BlurAncestralSampler", steps=3),
         sampler(f"{OLD}.blur_ancestral.BlurAncestralSampler", steps=3)),
    ),
    "simplex": (
        "sudoku_simplex",
        f"{REF}.simplex_diffusion.SimplexDiffusion",
        {},
        (sampler("core.samplers.simplex.SimplexBridgeSampler", steps=4),
         sampler(f"{OLD}.simplex.SimplexBridgeSampler", steps=4)),
    ),
    "spherical": (
        "sudoku_spherical",
        f"{REF}.spherical_flow_matching.SphericalFlowMatching",
        {},
        (sampler("core.samplers.spherical.SphericalFlowSampler", steps=4),
         sampler(f"{OLD}.spherical.SphericalFlowSampler", steps=4)),
    ),
    "uniform": (
        "uniform_diffusion",
        f"{REF}.uniform_diffusion.UniformDiffusion",
        {},
        (sampler("core.samplers.uniform_ancestral.UniformAncestralSampler", steps=8),
         sampler(f"{OLD}.uniform_ancestral.UniformAncestralSampler", steps=8)),
    ),
    "duo": (
        "duo",
        f"{REF}.duo.Duo",
        {},
        (sampler("core.samplers.uniform_ancestral.UniformAncestralSampler", steps=8),
         sampler(f"{OLD}.uniform_ancestral.UniformAncestralSampler", steps=8)),
    ),
    "d3pm": (
        "d3pm_absorb",
        f"{REF}.d3pm_absorb.D3PMAbsorb",
        {},
        (sampler("core.samplers.ancestral.AncestralSampler", steps=8),
         sampler(f"{OLD}.ancestral.AncestralSampler", steps=8)),
    ),
    "sedd": (
        "sedd_absorb",
        f"{REF}.sedd_absorb.SeddAbsorb",
        {},
        (sampler("core.samplers.analytic.AnalyticSampler", steps=8),
         sampler(f"{OLD}.analytic_sedd.AnalyticSampler", steps=8)),
    ),
    "replaid": (
        "replaid_sudoku_hard",
        f"{REF}.replaid.RePlaid",
        {},
        (sampler("core.samplers.gaussian_ddpm.GaussianDDPMSampler", steps=4),
         sampler(f"{OLD}.gaussian_ddpm.GaussianDDPMSampler", steps=4)),
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_old_family_checkpoint_converts_loads_and_samples(name):
    experiment, reference, overrides, (current_sampler, old_sampler) = CASES[name]
    torch.manual_seed(0)
    old = TrainingModule(config(experiment, overrides, reference))
    perturb(old)
    checkpoint = saved(old)
    torch.manual_seed(123)
    new = TrainingModule(config(experiment, overrides))
    converted = convert_checkpoint(checkpoint, new.config)
    new.load_state_dict(converted["state_dict"], strict=True)
    new.ema.load_state_dict(converted["ema"])

    plan = conversion_plan(new.config)
    old_state = old.state_dict()
    for key, value in new.state_dict().items():
        assert torch.equal(value, old_state[plan.names.get(key, key)]), key
    permutation = plan.permutation()
    for index, shadow in enumerate(new.ema.shadow_params):
        assert torch.equal(shadow, old.ema.shadow_params[permutation[index]])

    cfg = config(experiment, overrides)
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    old.algorithm.eval()
    new.algorithm.eval()
    torch.manual_seed(11)
    expected, expected_info = old.algorithm.generate(instantiate(old_sampler), prompt, fixed)
    torch.manual_seed(11)
    actual, actual_info = new.algorithm.generate(instantiate(current_sampler), prompt, fixed)
    assert torch.equal(expected, actual)
    assert expected_info["nfe_per_sample"] == actual_info["nfe_per_sample"]
