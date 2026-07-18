from __future__ import annotations

from collections.abc import Callable

from .base import Sampler


class SamplerRegistry:
    def __init__(self):
        self._factories: dict[str, Callable[[object], Sampler]] = {}

    def register(self, name: str):
        def decorator(factory):
            if name in self._factories:
                raise ValueError(f"Sampler already registered: {name}")
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, config) -> Sampler:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown sampler {name!r}; available samplers: "
                f"{', '.join(sorted(self._factories))}"
            ) from exc
        return factory(config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


sampler_registry = SamplerRegistry()


def create_sampler(config, *, default: str | None = None) -> Sampler:
    # Importing registers the built-in factories without making registry.py
    # depend on their implementations during module initialization.
    from . import (  # noqa: F401
        analytic,
        ancestral,
        ancestral_cached,
        autoregressive,
        autoregressive_cached,
        cadd,
        candi,
        candi_cached,
        continuous_embedding,
        continuous_onehot,
        permuted_autoregressive,
        permuted_autoregressive_uncached,
        semi_autoregressive,
        semicat_dopri5,
        semicat_euler,
        semicat_flow_map,
        flm_euler,
        sfm,
    )

    predictor = default or getattr(config.sampling, "predictor", None)
    if predictor is None:
        raise ValueError("sampling.predictor must select a registered sampler")
    return sampler_registry.create(predictor, config)
