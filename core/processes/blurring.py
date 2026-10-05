"""DCT corruption and exact reverse Gaussian bridge for local sequence spans."""

import math
import torch


def dct_matrix(length, *, device=None, dtype=torch.float32):
    if length < 1:
        raise ValueError("DCT length must be positive")
    n = torch.arange(length, device=device, dtype=dtype)[:, None]
    k = torch.arange(length, device=device, dtype=dtype)[None]
    matrix = torch.cos(math.pi * (n + 0.5) * k / length)
    scale = torch.full((length,), math.sqrt(2 / length), device=device, dtype=dtype)
    scale[0] = math.sqrt(1 / length)
    return matrix * scale


class BlurringProcess:
    def corrupt_state(self, clean, coefficients, corruption_mask):
        # The schedule encodes the active spans and leaves all other sites clean.
        return self.corrupt(clean, coefficients)

    def prior_state(self, clean, coefficients):
        return self.prior(clean)

    def __init__(self):
        self._basis_cache = {}

    def basis(self, length, reference):
        key = length, reference.device, reference.dtype
        if key not in self._basis_cache:
            self._basis_cache[key] = dct_matrix(
                length, device=reference.device, dtype=reference.dtype
            )
        return self._basis_cache[key]

    def corrupt(self, clean, path, *, noise=None):
        result = clean.clone()
        noise = torch.randn_like(clean) if noise is None else noise
        if noise.shape != clean.shape:
            raise ValueError("corruption noise must match clean state")
        for group in path.groups:
            length, spans = group.length, group.spans
            basis = self.basis(length, clean)
            values = torch.stack(
                [clean[batch, start:end] for batch, start, end in spans]
            )
            noises = torch.stack(
                [noise[batch, start:end] for batch, start, end in spans]
            )
            modes = torch.matmul(basis.T, values)
            mode_noise = torch.matmul(basis.T, noises)
            coefficients = group.coefficients
            updates = torch.matmul(
                basis,
                coefficients.attenuation * modes
                + coefficients.sigma[:, None, None] * mode_noise,
            )
            for update, (batch, start, end) in zip(updates, spans):
                result[batch, start:end] = update
        return result

    def prior(self, clean):
        return torch.randn_like(clean)

    def reverse_mode_moments(self, state_modes, clean_modes, current, target):
        tiny = torch.finfo(target.attenuation.dtype).tiny
        ratio = current.attenuation / target.attenuation.clamp_min(tiny)
        d2 = (
            current.sigma[:, None, None].square()
            - ratio.square() * target.sigma[:, None, None].square()
        )
        if d2.min() < -1e-6:
            raise RuntimeError("reverse transition has negative variance")
        d2 = d2.clamp_min(0)
        sigma_t2 = current.sigma[:, None, None].square().clamp_min(1e-20)
        mean = (
            target.sigma[:, None, None].square() / sigma_t2 * ratio * state_modes
            + d2 / sigma_t2 * target.attenuation * clean_modes
        )
        variance = d2 * target.sigma[:, None, None].square() / sigma_t2
        return mean, variance

    def reverse(self, state, clean, current_path, target_path, *, noise=None):
        if tuple((g.length, g.spans) for g in current_path.groups) != tuple(
            (g.length, g.spans) for g in target_path.groups
        ):
            raise ValueError("source and target blur paths must use identical spans")
        result = state.clone()
        noise = torch.randn_like(state) if noise is None else noise
        for current_group, target_group in zip(current_path.groups, target_path.groups):
            length, spans = current_group.length, current_group.spans
            basis = self.basis(length, state)
            values = torch.stack(
                [state[batch, start:end] for batch, start, end in spans]
            )
            estimates = torch.stack(
                [clean[batch, start:end] for batch, start, end in spans]
            )
            noises = torch.stack(
                [noise[batch, start:end] for batch, start, end in spans]
            )
            state_modes = torch.matmul(basis.T, values)
            clean_modes = torch.matmul(basis.T, estimates)
            noise_modes = torch.matmul(basis.T, noises)
            mean, variance = self.reverse_mode_moments(
                state_modes,
                clean_modes,
                current_group.coefficients,
                target_group.coefficients,
            )
            draw = (target_group.coefficients.sigma > 0).to(state)[:, None, None]
            updates = torch.matmul(basis, mean + draw * variance.sqrt() * noise_modes)
            for update, (batch, start, end) in zip(updates, spans):
                result[batch, start:end] = update
        return result
