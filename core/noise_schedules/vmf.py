"""Finite-concentration vMF schedule; framework time runs from data to noise."""

import torch


class VMFSchedule:
    output_kind = "concentration"

    def __init__(self, kappa_max=250.0):
        if kappa_max <= 0:
            raise ValueError("kappa_max must be positive")
        self.kappa_max = float(kappa_max)

    def validate_process(self, process):
        if self.kappa_max > process.kappa_max:
            raise ValueError("schedule exceeds vMF table concentration")

    def __call__(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("vMF time must be in [0, 1]")
        return self.kappa_max * (1 - time), torch.full_like(time, -self.kappa_max)
