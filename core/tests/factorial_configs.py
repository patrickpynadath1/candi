"""Sudoku corruption-family configs for model tests (one content view each).

Composes the same cells the retired corruption x self-conditioning example
built, without its launcher: six corruption families, self-conditioning off or
on, and the three 128-NFE sampler policies it evaluated.
"""

from omegaconf import OmegaConf

from core.sweeps.configuration import compose

FAMILIES = ("blurring", "gaussian", "spherical", "simplex", "masked", "uniform")


def factorial_config(family):
    cfg = compose(
        "corruption_factorial",
        {"hybrid_view@model.views.content": family, "data.difficulty": "hard"},
    )
    view = cfg.model.views.content
    if family == "gaussian":
        schedule = compose(overrides={"noise_schedule": "cdcd_gaussian_adaptive"})
        base = compose(overrides={"noise_schedule": "one_hot_rank"}).noise_schedule
        base.inversion = "analytic"
        base.coupling = "linear"
        schedule.noise_schedule.base = base
        view.schedule = OmegaConf.to_container(schedule.noise_schedule, resolve=False)
        view.representation = {
            "_target_": "core.representations.noisy_inputs.ProjectedRepresentation",
            "state_dim": "${task.vocab_size}",
            "hidden_size": "${model.backbone.hidden_size}",
            "state_representation": {
                "_target_": "core.representations.one_hot.OneHotRepresentation",
                "vocab_size": "${task.vocab_size}",
            },
        }
    elif family == "spherical":
        # Native default, not the adaptive/truncated historical Sudoku preset.
        source = compose("spherical_flow_matching")
        view.schedule = OmegaConf.to_container(source.noise_schedule, resolve=False)
        view.schedule.dimension = source.representation.dimension
    elif family == "blurring":
        view.schedule.feature_rho = 0.5
    return cfg


def cells():
    """(family, self_conditioning, config) for every factorial cell."""
    for family in FAMILIES:
        base = factorial_config(family)
        for enabled in (False, True):
            cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
            cfg.self_conditioning.enabled = enabled
            yield family, enabled, cfg


def samplers(entropy_bound=0.1):
    """The native, native-greedy and entropy-bounded 128-NFE policies."""
    choices = [
        ("hybrid", {"steps": 127, "final_denoise": "greedy", "final_transition_noise": False}),
        (
            "hybrid",
            {
                "steps": 127,
                "final_denoise": "greedy",
                "final_transition_noise": False,
                "greedy_clean_estimate": True,
            },
        ),
        (
            "parallel_entropy_bounded",
            {
                "steps": 127,
                "entropy_bound": entropy_bound,
                "freeze": True,
                "greedy_clean_estimate": True,
            },
        ),
    ]
    for group, overrides in choices:
        sampler = compose(overrides={"sampler": group}).sampler
        for key, value in overrides.items():
            OmegaConf.update(sampler, key, value)
        yield group, sampler
