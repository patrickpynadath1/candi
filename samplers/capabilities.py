"""Runtime method contracts advertised by algorithm classes."""

CAPABILITY_METHODS: dict[str, tuple[str, ...]] = {
    "next_token_logits": ("next_token_logits",),
    "discrete_prior": ("prior_sample",),
    "continuous_prior": ("prior_sample",),
    "noise_schedule": ("noise",),
    "ancestral_transition": ("sample_ancestral_transition",),
    "analytic_transition": ("sample_analytic_transition",),
    "denoiser_transition": ("sample_denoiser_transition",),
    "greedy_denoise": ("sample_greedy_denoise",),
    "zero_time_denoise": ("sample_zero_time_denoise",),
    "continuous_schedule": ("sampling_continuous_schedule",),
    "continuous_transition": ("sample_continuous_transition",),
    "embedding_projection": ("sampling_embeddings",),
    "cadd_continuous_transition": ("sample_cadd_continuous_transition",),
    "cadd_discrete_transition": ("sample_cadd_discrete_transition",),
    "candi_continuous_transition": ("sample_candi_continuous_transition",),
    "candi_discrete_transition": ("sample_candi_discrete_transition",),
    "candi_cached_continuous_transition": (
        "sample_candi_cached_continuous_transition",
    ),
    "candi_cached_discrete_transition": (
        "sample_candi_cached_discrete_transition",
    ),
    "semicat_flow_map": ("sample_semicat_flow_map",),
    "semicat_vector_field": ("sample_semicat_vector_field",),
}
