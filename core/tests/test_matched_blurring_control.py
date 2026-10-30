import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from core.sweeps.configuration import compose
from core.processes.gaussian import GaussianProcess
from core.processes.blurring import BlurringProcess
from core.noise_schedules.blurring import BlurCoefficients
from core.noise_schedules.gaussian_path import GaussianPath


def test_gaussian_control_uses_real_gaussian_and_identical_model():
    blur = compose("tinygsm_blurring_sc_adaptive")
    gaussian = compose("tinygsm_gaussian_matched_blurring_sc_adaptive")
    for key in [
        "model",
        "representation",
        "prediction",
        "self_conditioning",
        "objective",
        "time_sampler",
        "parameterization",
        "loader",
        "optimizer",
        "training",
        "trainer",
        "data",
        "task",
    ]:
        assert OmegaConf.to_container(
            blur[key], resolve=True
        ) == OmegaConf.to_container(gaussian[key], resolve=True), key
    assert (
        gaussian.algorithm._target_
        == "core.algorithms.gaussian_diffusion.GaussianDiffusion"
    )
    assert gaussian.process._target_ == "core.processes.gaussian.GaussianProcess"
    assert (
        gaussian.sampler._target_ == "core.samplers.gaussian_ddpm.GaussianDDPMSampler"
    )
    assert gaussian.sampler.steps == blur.sampler.steps == 512
    assert gaussian.sampler.epsilon == 0 and gaussian.sampler.prompt_mode == "clean"
    assert (
        gaussian.sampler.self_conditioning_mode == blur.sampler.self_conditioning_mode
    )


def test_gaussian_control_learns_the_codebook_and_samples():
    cfg = compose("tinygsm_gaussian_matched_blurring_sc_adaptive")
    cfg.task.vocab_size = 11
    cfg.task.length = 7
    cfg.model.hidden_size = 16
    cfg.model.cond_dim = 8
    cfg.model.n_blocks = 1
    cfg.model.n_heads = 2
    cfg.model.output_dim = 8
    cfg.representation.dimension = 8
    cfg.noise_schedule.grid_size = 65
    cfg.noise_schedule.quadrature_order = 16
    algorithm = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 11, (2, 7))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    result = algorithm.loss(tokens, mask, training=True, step=1)
    result["loss"].backward()
    grad = algorithm.representation.table.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert algorithm.time_sampler.count.item() == 2
    cfg.sampler.steps = 4
    algorithm.eval()
    fixed = ~mask
    fixed[:, :2] = True
    output, meta = instantiate(cfg.sampler).sample(algorithm, tokens, fixed)
    assert torch.equal(output[:, :2], tokens[:, :2]) and meta["nfe"] == 5


def test_gaussian_and_blur_samplers_use_same_reverse_bridge():
    state = torch.randn(2, 5, 3, dtype=torch.float64)
    clean = torch.randn_like(state)
    noise = torch.randn_like(state)
    alpha_t = torch.tensor([0.2, 0.4], dtype=torch.float64)
    sigma_t = 1 - alpha_t
    alpha_s = alpha_t + 0.1
    sigma_s = 1 - alpha_s
    t = GaussianPath(alpha_t, sigma_t, alpha_t * 0, alpha_t * 0)
    s = GaussianPath(alpha_s, sigma_s, alpha_s * 0, alpha_s * 0)

    def coeff(path):
        attenuation = path.alpha[:, None, None].expand_as(state)
        return BlurCoefficients(
            attenuation,
            path.sigma,
            torch.zeros_like(state),
            torch.zeros_like(path.sigma),
        )

    mean, var = BlurringProcess().reverse_mode_moments(state, clean, coeff(t), coeff(s))
    actual = GaussianProcess().reverse_posterior(state, clean, t, s, noise)
    torch.testing.assert_close(
        actual, mean + var.sqrt() * noise, atol=1e-12, rtol=1e-12
    )
