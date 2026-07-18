from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from .capabilities import CAPABILITY_METHODS


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    compatible_families: frozenset[str]
    required_capabilities: frozenset[str]


@dataclass(frozen=True)
class SamplingRequest:
    num_samples: int
    num_steps: int | None = None
    eps: float = 1e-5
    prompt_tokens: torch.Tensor | None = None
    prompt_mask: torch.Tensor | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingResult:
    samples: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class Sampler(abc.ABC):
    spec: ClassVar[SamplerSpec]

    def validate_model(self, model) -> None:
        family = getattr(model, "sampling_family", None)
        if family not in self.spec.compatible_families:
            raise ValueError(
                f'Sampler "{self.spec.name}" is not compatible with sampling '
                f'family {family!r}; expected one of '
                f"{sorted(self.spec.compatible_families)}"
            )
        available = set(getattr(model, "sampling_capabilities", ()))
        missing = self.spec.required_capabilities - available
        if missing:
            raise ValueError(
                f'Sampler "{self.spec.name}" cannot use '
                f"{type(model).__name__}; missing capabilities: "
                f"{', '.join(sorted(missing))}"
            )
        missing_methods = {
            method
            for capability in self.spec.required_capabilities
            for method in CAPABILITY_METHODS.get(capability, ())
            if not callable(getattr(model, method, None))
        }
        if missing_methods:
            raise TypeError(
                f'{type(model).__name__} advertises capabilities required by '
                f'"{self.spec.name}" but does not implement methods: '
                f"{', '.join(sorted(missing_methods))}"
            )

    @abc.abstractmethod
    def sample(self, model, request: SamplingRequest) -> SamplingResult:
        raise NotImplementedError
