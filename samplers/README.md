# Sampling architecture

Sampling trajectories live in this package. Algorithm classes in `algos/`
retain training losses and mathematical transition operations, but do not own
generation loops.

Every sampler declares:

- `compatible_families`: the semantic algorithm families it supports;
- `required_capabilities`: the operations it needs from an algorithm;
- a registry name used by `sampling.predictor` or an algorithm's
  `default_sampler`.

`capabilities.py` maps capability names to concrete method contracts. The
runner validates both the declaration and the methods before generation.

| Sampler | Compatible family | Key required operations |
| --- | --- | --- |
| `ar` | `autoregressive` | `next_token_logits` |
| `ar_cached` | `autoregressive` | cached next-token logits and causal KV cache |
| `permuted_ar` | `permuted_autoregressive` | cached logits with decoupled query/key positions |
| `permuted_ar_uncached` | `permuted_autoregressive` | full-context logits with decoupled query/key positions |
| `ancestral`, `ancestral_cache` | absorbing/uniform diffusion | prior, noise schedule, ancestral transition |
| `entropy_bounded` | absorbing diffusion | prior and clean-token log probabilities |
| `uniform_entropy_bounded` | uniform diffusion | clean prediction and reverse transition |
| `candi_entropy_bounded` | CANDI | continuous trajectory and clean-token probabilities |
| `analytic` | absorbing/uniform diffusion | prior, noise schedule, analytic transition |
| `continuous_onehot` | continuous one-hot | prior, inference schedule, continuous transition |
| `continuous_embedding` | continuous embedding | prior, inference schedule, continuous transition |
| `cadd` | `cadd` | continuous and discrete CADD transitions |
| `candi` | `candi` | uncached continuous and discrete transitions |
| `candi_cached` | `candi` | embedding cache and cached transitions |
| `semicat_flow_map` | `semicat` | flow map |
| `semicat_euler` | `semicat` | vector field |
| `semicat_dopri5` | `semicat` | vector field and adaptive ODE integration |
| `latent_fm` | `latent_fm` | hidden-state prior and decoder/re-encoder flow map |
| `replaid_ddpm` | `replaid` | Gaussian prior and categorical clean-embedding prediction |
| `replaid_keep_ddpm` | `replaid_keep` | Gaussian trajectory with progressive hard clean-token reveals |
| `replaid_keep_entropy_bounded` | `replaid_keep` | Gaussian trajectory with entropy-bounded hard embedding reveals |
| `semi_ar` | absorbing/uniform diffusion | ancestral transition and zero-time denoising |

To add a sampler:

1. Implement `Sampler` and a `SamplerSpec`.
2. Register a config factory with `sampler_registry.register`.
3. Add capability-to-method mappings in `capabilities.py`.
4. Declare the family and capabilities on compatible algorithms.
5. Add contract tests using a minimal fake model.

The public `TrainerBase.generate_samples` method remains as a compatibility
facade. It creates a `SamplingRequest`, resolves the sampler, validates the
model contract, and returns the generated token tensor.
