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
        "blurring_ode",
        "block_diffusion",
        "block_diffusion_euler",
        "cadd",
        "candi",
        "candi_imle",
        "candi_cached",
        "candi_looped",
        "candi_looped_entropy_bounded",
            "candi_embedding",
            "candi_embedding_entropy_bounded",
            "candi_entropy_bounded",
        "continuous_embedding",
        "continuous_onehot",
        "entropy_bounded",
        "semi_ar",
        "permuted_ar",
        "permuted_ar_uncached",
        "semicat_dopri5",
        "semicat_euler",
        "semicat_flow_map",
        "flm_euler",
        "frm_euler",
        "frm_held",
        "latent_fm",
        "replaid_ddpm",
        "replaid_hybrid_ddpm",
        "replaid_hybrid_entropy_bounded",
        "replaid_keep_ddpm",
        "replaid_keep_entropy_bounded",
        "renoising",
        "sfm",
        "simplex",
        "paper_discrete",
        "uniform_entropy_bounded",
        "uniform_entropy_bounded_renoise",
    }


def test_removed_aliases_are_not_registered():
    assert "posterior" not in sampler_registry.names()
    assert "semicat" not in sampler_registry.names()
