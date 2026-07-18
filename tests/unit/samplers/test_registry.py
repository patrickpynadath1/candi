from types import SimpleNamespace

from samplers.registry import create_sampler, sampler_registry


def test_every_concrete_sampler_has_a_unique_registry_name():
    create_sampler(
        SimpleNamespace(
            sampling=SimpleNamespace(predictor="ar", use_float64=False),
            algo=None,
        )
    )

    assert set(sampler_registry.names()) == {
        "analytic",
        "ancestral",
        "ancestral_cache",
        "ar",
        "ar_cached",
        "cadd",
        "candi",
        "candi_cached",
        "continuous_embedding",
        "continuous_onehot",
        "semi_ar",
        "permuted_ar",
        "permuted_ar_uncached",
        "semicat_dopri5",
        "semicat_euler",
        "semicat_flow_map",
        "flm_euler",
        "sfm",
    }


def test_removed_aliases_are_not_registered():
    assert "posterior" not in sampler_registry.names()
    assert "semicat" not in sampler_registry.names()
