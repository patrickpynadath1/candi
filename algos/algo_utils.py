# generic function utils for algos
import torch
from torch.distributions import Normal

def sigma_from_time_vectorized(
    t: torch.Tensor, sigmas: torch.Tensor, errors: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized version of sigma_from_time using linear interpolation.

    Args:
        t: tensor of target Bayes error rates (ε ∈ [0, 1])
        sigmas: 1D tensor of σ values
        errors: 1D tensor of corresponding ε(σ)

    Returns:
        Tensor of interpolated σ values
    """
    t = t.clamp(min=errors[0].item(), max=errors[-1].item())
    t = t.to(sigmas.device)

    # Normalize ε ∈ [0, 1] to indices in [0, len(errors)-1]
    indices = torch.searchsorted(errors, t, right=True).clamp(1, len(errors) - 1)
    i0 = indices - 1
    i1 = indices

    e0 = errors[i0]
    e1 = errors[i1]
    s0 = sigmas[i0]
    s1 = sigmas[i1]

    interp_t = (t - e0) / (e1 - e0 + 1e-8)
    return s0 + interp_t * (s1 - s0)



def sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def expected_rank(d: int, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Vectorized expected rank of the true coordinate in a one-hot vector
    corrupted by Gaussian noise.

    Args:
        d: dimension (number of coordinates)
        a: amplitude(s) of the true coordinate, scalar or tensor
        sigma: noise standard deviation(s), scalar or tensor

    Returns:
        Expected rank(s), same broadcasted shape as a and sigma
    """
    # ensure tensors
    a = torch.as_tensor(a, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)

    # standard normal CDF in torch
    Phi = torch.distributions.Normal(0.0, 1.0).cdf

    # compute p = P(other beats true coordinate)
    p = Phi(-a / (sigma * torch.sqrt(torch.tensor(2.0, dtype=torch.float32))))

    return (d - 1) * p / d


# below is the noise schedule used for embedding diffusion -- follows the VE SDE schedule
def training_sigma_ve(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """
    Map training time t ∈ [0,1] to sigma via log-linear schedule.

    σ(t) = σ_min * (σ_max / σ_min) ** t

    Args:
        t: tensor of shape [...] with values in [0,1]
        sigma_min: minimum noise level
        sigma_max: maximum noise level

    Returns:
        σ(t) with same shape as t
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    log_ratio = torch.log(torch.tensor(sigma_max / sigma_min, dtype=torch.float32).to(t.device))
    return sigma_min * torch.exp(t * log_ratio)


def inference_sigmas(n_steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """
    Log-linear schedule of sigmas for inference, decreasing from sigma_max to sigma_min.

    σ_k = σ_max * (σ_min / σ_max) ** (k / (n_steps-1))

    Args:
        n_steps: number of inference steps
        sigma_min: minimum noise level
        sigma_max: maximum noise level

    Returns:
        Tensor of shape [n_steps], from σ_max down to σ_min
    """
    ks = torch.linspace(0, 1, n_steps, dtype=torch.float32)
    return sigma_max * (sigma_min / sigma_max) ** ks


def gaussian_to_disc_corruption(K: int, sigma: float) -> float:
    normal = Normal(0., 1.)
    s = torch.linspace(-4*sigma, 1+6*sigma, 1001)
    phi = normal.log_prob((s-1)/sigma).exp() / sigma
    cdf_power = normal.cdf(s/sigma).pow(K-1)
    return 1.0 - torch.trapz(cdf_power * phi, s).item()


def build_error_to_sigma_schedule(
    vocab_size: int,
    sigma_range: tuple = (0.01, 1.0),
    num_points: int = 500,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = torch.linspace(*sigma_range, steps=num_points, device=device)
    errors = torch.tensor(
        [gaussian_to_disc_corruption(vocab_size, s.item()) for s in sigmas], 
        device=device
    )
    return sigmas, errors


def error_to_sigma(
    error_rates: torch.Tensor,
    sigmas: torch.Tensor,
    errors: torch.Tensor
) -> torch.Tensor:
    error_rates = error_rates.clamp(min=errors[0].item(), max=errors[-1].item())
    
    # Find bracketing indices
    indices = torch.searchsorted(errors, error_rates, right=True).clamp(1, len(errors)-1)
    i0 = indices - 1
    i1 = indices
    
    # Get bracketing values
    e0 = errors[i0]
    e1 = errors[i1]
    s0 = sigmas[i0]
    s1 = sigmas[i1]
    
    # Linear interpolation
    interp_t = (error_rates - e0) / (e1 - e0 + 1e-8)
    return s0 + interp_t * (s1 - s0)


def sigma_to_error(
    sigma_vals: torch.Tensor,
    sigmas: torch.Tensor,
    errors: torch.Tensor
) -> torch.Tensor:
    sigma_vals = sigma_vals.clamp(min=sigmas[0], max=sigmas[-1])
    
    # Find bracketing indices
    indices = torch.searchsorted(sigmas, sigma_vals, right=True).clamp(1, len(sigmas)-1)
    i0 = indices - 1
    i1 = indices
    
    # Get bracketing values
    s0 = sigmas[i0]
    s1 = sigmas[i1]
    e0 = errors[i0]
    e1 = errors[i1]
    
    # Linear interpolation
    interp_t = (sigma_vals - s0) / (s1 - s0 + 1e-8)
    return e0 + interp_t * (e1 - e0)
