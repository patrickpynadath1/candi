import torch

from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView
from core.samplers.hybrid_updates import (
    DiscretePosteriorUpdate,
    GaussianPosteriorUpdate,
)
from core.modifiers.hybrid.time_maps import IdentityTimeMap, PowerTimeMap
from core.models.dit import DiT
from core.modifiers.hybrid.model import HybridModel
from core.representations.noisy_inputs import (
    TokenRepresentation,
    ProjectedRepresentation,
)
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.noise_schedules.candi_rank import CANDIRankSchedule
from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.noise_schedules.log_linear import LogLinear
from core.objectives.cross_entropy import CrossEntropy
from core.parameterizations.posterior import PosteriorParameterization
from core.processes.candi_masking import CANDIMaskingProcess
from core.processes.candi_gaussian import CANDIGaussianProcess
from core.processes.gaussian import GaussianProcess
from core.representations.one_hot import OneHotRepresentation
from core.samplers.hybrid import HybridSampler
from core.samplers.hybrid_entropy_bounded import CANDIEntropyBoundedSampler
from core.time_sampling.uniform import UniformTimeSampler


def make_algorithm(
    *, self_conditioning=False, second_process=None, second_schedule=None, time_map=None
):
    vocab, hidden = 5, 16
    discrete = DiffusionView(
        CANDIMaskingProcess(vocab),
        LogLinear(0),
        TokenRepresentation(vocab + 1, hidden, 0.5),
        DiscretePosteriorUpdate(),
        IdentityTimeMap(),
    )
    gaussian = DiffusionView(
        second_process or GaussianProcess(),
        second_schedule or LinearInterpolationSchedule(),
        ProjectedRepresentation(
            vocab,
            hidden,
            0.5,
            bias=False,
            state_representation=OneHotRepresentation(vocab),
        ),
        GaussianPosteriorUpdate(),
        time_map or IdentityTimeMap(),
    )
    backbone = DiT(vocab, hidden, 8, 1, 4, dropout=0, adaLN=False)
    model = HybridModel(
        backbone, [discrete, gaussian], self_conditioning=self_conditioning
    )
    prediction = PredictionCoordinator(
        enabled=self_conditioning, probability=1, evaluation_probability=1
    )
    return HybridDiffusion(
        model,
        PosteriorParameterization(),
        UniformTimeSampler(1e-3, False),
        CrossEntropy(),
        length=7,
        prediction=prediction,
    )


def test_parallel_loss_corrupts_two_views_and_backpropagates():
    torch.manual_seed(1)
    algo = make_algorithm()
    tokens = torch.randint(0, 5, (3, 7))
    result = algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert algo.model.backbone.output_layer.linear.weight.grad is not None


def test_local_time_maps_are_applied_per_branch():
    algo = make_algorithm(time_map=PowerTimeMap(2))
    time = torch.tensor([0.2, 0.7])
    points = algo.points(time)
    assert torch.equal(points[0].time, time)
    assert torch.allclose(points[1].time, time.square())


def test_joint_sampler_uses_one_shared_model_evaluation_per_step():
    algo = make_algorithm()
    calls = 0
    original = algo.model.backbone.forward_hidden

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    algo.model.backbone.forward_hidden = counted
    prompt = torch.zeros(2, 7, dtype=torch.long)
    mask = torch.zeros_like(prompt, dtype=torch.bool)
    result, metadata = HybridSampler(steps=3).sample(algo, prompt, mask)
    assert result.shape == prompt.shape
    assert calls == metadata["nfe"] == 4


def test_sampling_self_conditioning_has_one_fused_feedback_vector():
    algo = make_algorithm(self_conditioning=True)
    prompt = torch.zeros(2, 7, dtype=torch.long)
    result, metadata = HybridSampler(steps=2).sample(
        algo, prompt, torch.zeros_like(prompt, dtype=torch.bool)
    )
    assert result.shape == prompt.shape
    assert metadata["nfe"] == 3


def test_candi_rank_schedule_and_historical_step_formula():
    schedule = CANDIRankSchedule(12, 0.05, 0.4)
    source = schedule(torch.tensor([1.0]))
    destination = schedule(torch.tensor([0.5]))
    assert source.sigma > destination.sigma
    state = torch.ones(1, 2, 12)
    clean = torch.zeros_like(state)
    actual = CANDIGaussianProcess().reverse_posterior(
        state, clean, source, destination, torch.randn_like(state)
    )
    expected = state - (source.sigma - destination.sigma)[:, None, None] * (
        state / destination.sigma.square()[:, None, None]
    )
    assert torch.equal(actual, expected)


def test_candi_training_corruption_is_exact_masking_without_accidental_keeps():
    process = CANDIMaskingProcess(mask_index=5)
    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    state = process.corrupt(
        tokens, torch.tensor([0.0]), torch.ones_like(tokens, dtype=torch.bool)
    )
    assert torch.equal(state, torch.full_like(tokens, 5))


def test_alternate_absorbing_plus_standard_gaussian_composition():
    algo = make_algorithm(
        second_process=GaussianProcess(),
        second_schedule=LinearInterpolationSchedule(),
        time_map=PowerTimeMap(0.5),
    )
    prompt = torch.randint(0, 5, (1, 7))
    fixed = torch.tensor([[True, True, False, False, False, False, False]])
    result, _ = HybridSampler(steps=2).sample(algo, prompt, fixed)
    assert torch.equal(result[fixed], prompt[fixed])


def test_hybrid_entropy_bounded_accepts_without_discrete_branch_updates():
    algo = make_algorithm()
    prompt = torch.zeros(2, 7, dtype=torch.long)
    result, metadata = CANDIEntropyBoundedSampler(
        entropy_bound=100.0, steps=1, greedy_clean_estimate=True
    ).sample(algo, prompt, torch.zeros_like(prompt, dtype=torch.bool))
    assert result.shape == prompt.shape
    assert metadata["nfe"] == 1
    assert all(value == metadata["nfe"] for value in metadata["nfe_per_sample"])
