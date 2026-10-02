"""Length-local separable heat-kernel schedule for blurring diffusion."""

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class BlurCoefficients:
    attenuation: torch.Tensor
    sigma: torch.Tensor
    dattenuation_dt: torch.Tensor
    dsigma_dt: torch.Tensor


@dataclass(frozen=True)
class BlurSpanGroup:
    length: int
    spans: tuple
    coefficients: BlurCoefficients


@dataclass(frozen=True)
class BlurLayoutPath:
    groups: tuple


class BlurringSchedule:
    def __init__(
        self,
        basis="dct",
        feature_rho=0.5,
        feature_lambda_min=0.25,
        feature_lambda_max=1e9,
        feature_tau_min=1e-9,
        feature_tau_max=20.0,
        time_warp="linearized_crossings",
        linearized_crossing_min=0.05,
        linearized_crossing_max=0.95,
        tau_min_attenuation=0.01,
        terminal_nonzero_mode=0.01,
        terminal_mean=0.03,
        lambda_scale=1.0,
        noise_power=2.0,
        crossing_max_decile_fraction=0.15,
    ):
        if basis not in {"dct", "isotropic"}:
            raise ValueError("basis must be dct or isotropic")
        if not 0 <= feature_rho <= 1:
            raise ValueError("feature_rho must lie in [0, 1]")
        if not 0 < feature_lambda_min <= feature_lambda_max:
            raise ValueError("feature lambda bounds must be positive and ordered")
        if not 0 < feature_tau_min < feature_tau_max:
            raise ValueError("feature tau bounds must be positive and ordered")
        if time_warp not in {"none", "linearized_crossings"}:
            raise ValueError("time_warp must be none or linearized_crossings")
        if not 0 < linearized_crossing_min < linearized_crossing_max < 1:
            raise ValueError("crossing bounds must lie strictly inside (0, 1)")
        if not 0 < terminal_mean < 1 or not 0 < terminal_nonzero_mode < 1:
            raise ValueError("terminal attenuations must lie in (0, 1)")
        if noise_power <= 0:
            raise ValueError("noise_power must be positive")
        self.basis, self.feature_rho = basis, float(feature_rho)
        self.feature_lambda_min, self.feature_lambda_max = (
            feature_lambda_min,
            feature_lambda_max,
        )
        self.feature_tau_min, self.feature_tau_max = feature_tau_min, feature_tau_max
        self.time_warp = time_warp
        self.linearized_crossing_min, self.linearized_crossing_max = (
            linearized_crossing_min,
            linearized_crossing_max,
        )
        self.tau_min_attenuation = tau_min_attenuation
        self.terminal_nonzero_mode, self.terminal_mean = (
            terminal_nonzero_mode,
            terminal_mean,
        )
        self.lambda_scale, self.noise_power = lambda_scale, noise_power
        self.crossing_max_decile_fraction = crossing_max_decile_fraction
        self._warp_cache = {}
        self._runtime_warp_cache = {}
        self._validated_shapes = set()

    def _constants(self, length):
        # A singleton uses the length-two clock calibration and its zero spatial
        # mode. This is the finite scalar continuation of the source equations;
        # all lengths >=2 retain the exact source calibration.
        calibration_length = max(length, 2)
        k = torch.arange(calibration_length, dtype=torch.float64)
        laplacian = 2 * (1 - torch.cos(math.pi * k / calibration_length))
        tau_min = self.tau_min_attenuation / (float(laplacian[-1]) * self.lambda_scale)
        tau_end = -math.log(self.terminal_nonzero_mode) / (
            float(laplacian[1]) * self.lambda_scale
        )
        return tau_min, tau_end + tau_min, -math.log(self.terminal_mean) / tau_end

    def _tau(self, time, length):
        low, high, _ = self._constants(length)
        log_ratio = math.log(high / low)
        unshifted = low * torch.exp(log_ratio * time)
        return unshifted - low, unshifted * log_ratio

    def _rates(self, length, time):
        _, _, floor = self._constants(length)
        if self.basis == "isotropic":
            return torch.full((length,), floor, device=time.device, dtype=time.dtype)
        k = torch.arange(length, device=time.device, dtype=time.dtype)
        return floor + self.lambda_scale * 2 * (1 - torch.cos(math.pi * k / length))

    def _base(self, time, length):
        tau, dtau = self._tau(time, length)
        rates = self._rates(length, time)
        attenuation = torch.exp(-tau[:, None] * rates)
        derivative = -dtau[:, None] * rates * attenuation
        sigma = time.clamp_min(0).pow(self.noise_power)
        dsigma = self.noise_power * time.clamp_min(0).pow(self.noise_power - 1)
        return attenuation, sigma, derivative, dsigma

    @staticmethod
    def _crossings(attenuation, sigma, grid):
        tiny = torch.finfo(grid.dtype).tiny
        log_snr = 2 * (
            attenuation.clamp_min(tiny).log() - sigma[:, None].clamp_min(tiny).log()
        )
        right = (log_snr <= 0).to(torch.int8).argmax(0)
        left = (right - 1).clamp_min(0)
        modes = torch.arange(attenuation.shape[1])
        fraction = (
            -log_snr[left, modes] / (log_snr[right, modes] - log_snr[left, modes])
        ).clamp(0, 1)
        return grid[left] + fraction * (grid[right] - grid[left])

    def _warp_knots(self, length):
        if length in self._warp_cache:
            return self._warp_cache[length]
        grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
        attenuation, sigma, _, _ = self._base(grid, length)
        crossings = self._crossings(attenuation, sigma, grid)
        target = torch.linspace(
            self.linearized_crossing_max,
            self.linearized_crossing_min,
            length,
            dtype=torch.float64,
        )
        inputs = torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                target.flip(0),
                torch.ones(1, dtype=torch.float64),
            )
        )
        outputs = torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                crossings.flip(0),
                torch.ones(1, dtype=torch.float64),
            )
        )
        self._warp_cache[length] = inputs, outputs
        return inputs, outputs

    def warp_time(self, time, length):
        if self.time_warp == "none":
            return time, torch.ones_like(time)
        key = length, time.device, time.dtype
        if key not in self._runtime_warp_cache:
            self._runtime_warp_cache[key] = tuple(
                value.to(time) for value in self._warp_knots(length)
            )
        inputs, outputs = self._runtime_warp_cache[key]
        indices = (torch.searchsorted(inputs, time.contiguous(), right=True) - 1).clamp(
            0, len(inputs) - 2
        )
        slope = (outputs[indices + 1] - outputs[indices]) / (
            inputs[indices + 1] - inputs[indices]
        )
        return outputs[indices] + slope * (time - inputs[indices]), slope

    def mode_coefficients(self, time, length):
        warped, warp_derivative = self.warp_time(time, length)
        attenuation, sigma, derivative, dsigma = self._base(warped, length)
        return (
            attenuation,
            sigma,
            derivative * warp_derivative[:, None],
            dsigma * warp_derivative,
        )

    def feature_rates(self, width, time):
        return torch.logspace(
            math.log10(self.feature_lambda_min),
            math.log10(self.feature_lambda_max),
            width,
            device=time.device,
            dtype=time.dtype,
        )

    def coefficients(self, time, length, width):
        warped, warp_derivative = self.warp_time(time, length)
        positional, sigma, _, dsigma = self._base(warped, length)
        tau, dtau = self._tau(warped, length)
        rates = self._rates(length, time)
        # Compute powers and their derivatives from the log attenuation. This
        # is algebraically identical to rho*a**(rho-1)*a' but remains finite
        # after high-frequency attenuation underflows to zero.
        pos_exponent = -self.feature_rho * tau[:, None] * rates
        pos = torch.exp(pos_exponent)
        dpos = (
            -self.feature_rho * dtau[:, None] * rates * pos * warp_derivative[:, None]
        )
        log_ratio = math.log(self.feature_tau_max / self.feature_tau_min)
        unshifted = self.feature_tau_min * torch.exp(log_ratio * time)
        feature_tau = unshifted - self.feature_tau_min
        feature_dtau = unshifted * log_ratio
        feature_rates = self.feature_rates(width, time)
        weight = 1 - self.feature_rho
        feature = torch.exp(-weight * feature_tau[:, None] * feature_rates)
        dfeature = -weight * feature_dtau[:, None] * feature_rates * feature
        joint = pos[:, :, None] * feature[:, None, :]
        derivative = (
            dpos[:, :, None] * feature[:, None, :]
            + pos[:, :, None] * dfeature[:, None, :]
        )
        return BlurCoefficients(joint, sigma, derivative, dsigma * warp_derivative)

    def training_time(self, time):
        return time

    def for_layout(self, time, mask, width, sequence_layout):
        grouped = {}
        for span in sequence_layout.spans(mask):
            grouped.setdefault(span[2] - span[1], []).append(span)
        groups = []
        for length, spans in grouped.items():
            self.validate_shape(length, width)
            batches = torch.tensor([span[0] for span in spans], device=time.device)
            groups.append(
                BlurSpanGroup(
                    length,
                    tuple(spans),
                    self.coefficients(time[batches], length, width),
                )
            )
        return BlurLayoutPath(tuple(groups))

    @torch.no_grad()
    def validate_shape(self, length, width):
        """Validate the actual local span geometry once, never a padded length."""
        key = length, width
        if key in self._validated_shapes:
            return
        grid = torch.linspace(0, 1, 2001, dtype=torch.float64)
        coefficients = self.coefficients(grid, length, width)
        if not torch.isfinite(coefficients.dattenuation_dt).all():
            raise ValueError("blur attenuation derivatives must remain finite")
        if not torch.equal(
            coefficients.attenuation[0], torch.ones_like(coefficients.attenuation[0])
        ):
            raise ValueError("blur attenuation must start at one")
        tiny = torch.finfo(grid.dtype).tiny
        snr = coefficients.attenuation.square() / coefficients.sigma[
            :, None, None
        ].square().clamp_min(tiny)
        crossed = snr <= 1
        if not crossed.any(0).all():
            raise ValueError("every blur mode must cross SNR=1 by t=1")
        positional, positional_sigma, _, _ = self.mode_coefficients(grid, length)
        positional_snr = positional.square() / positional_sigma[
            :, None
        ].square().clamp_min(tiny)
        positional_crossed = positional_snr <= 1
        if not positional_crossed.any(0).all():
            raise ValueError("every positional mode must cross SNR=1 by t=1")
        positional_crossings = grid[positional_crossed.to(torch.int8).argmax(0)]
        if not torch.all(positional_crossings[:-1] >= positional_crossings[1:]):
            raise ValueError("positional mode crossings must decrease with frequency")
        if self.basis == "dct" and length >= 32:
            histogram = torch.histc(positional_crossings, bins=10, min=0, max=1)
            if float(histogram.max()) / length > self.crossing_max_decile_fraction:
                raise ValueError("positional blur mode deaths are too bunched")
        if self.feature_rho < 1:
            log_ratio = math.log(self.feature_tau_max / self.feature_tau_min)
            unshifted = self.feature_tau_min * torch.exp(log_ratio * grid)
            feature_tau = unshifted - self.feature_tau_min
            feature = torch.exp(
                -(1 - self.feature_rho)
                * feature_tau[:, None]
                * self.feature_rates(width, grid)
            )
            feature_snr = feature.square() / coefficients.sigma[
                :, None
            ].square().clamp_min(tiny)
            feature_crossed = feature_snr <= 1
            if not feature_crossed.any(0).all():
                raise ValueError("every feature column must cross SNR=1 by t=1")
            feature_crossings = grid[feature_crossed.to(torch.int8).argmax(0)]
            if not torch.all(feature_crossings[:-1] >= feature_crossings[1:]):
                raise ValueError("feature columns must die in ascending column order")
            feature_histogram = torch.histc(feature_crossings, bins=10, min=0, max=1)
            if width >= 10 and int(feature_histogram.max()) > math.ceil(0.3 * width):
                raise ValueError("feature blur deaths are too bunched")
        ratio = coefficients.attenuation[1:] / coefficients.attenuation[:-1].clamp_min(
            tiny
        )
        d2 = (
            coefficients.sigma[1:, None, None].square()
            - ratio.square() * coefficients.sigma[:-1, None, None].square()
        )
        if d2.min() < -1e-10:
            raise ValueError("two-axis blur schedule violates Markov variance")
        if float(coefficients.attenuation[-1].square().max()) * length >= 0.1:
            raise ValueError(
                "terminal local-span signal leakage exceeds the configured bound"
            )
        self._validated_shapes.add(key)
