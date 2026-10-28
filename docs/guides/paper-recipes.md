# Configurable paper recipes

Select a recipe through the existing experiment group; tasks remain independent:

```bash
# Inspect the fully composed config without launching training.
uv run --locked python main.py +experiment=cadd task=text8 --cfg job
# Train after choosing the budget and dataset settings.
uv run --locked python main.py +experiment=mdlm task=text8 trainer.max_steps=20000
```

These presets compose existing components. They do not certify paper benchmark
parity. Backbone, tokenizer, training budget and inference budget must be matched
separately for a reproduction. Existing experiments retain their existing
settings (some have long reference budgets); inspect `--cfg job` before launching.
No preset is a request to submit a job to Ray.

## Information paths and denoising tasks

| Experiment | Construction and limits |
| --- | --- |
| `mdlm` | [MDLM](https://arxiv.org/abs/2406.07524): absorbing posterior, weighted CE, ancestral inference. |
| `md4` | [MD4](https://arxiv.org/abs/2406.04329): standard state-independent masking; shares MDLM components. Not GenMD4. |
| `radd` | [RADD](https://arxiv.org/abs/2406.03736): time-independent posterior. No unchanged-state prediction cache. |
| `d3pm_absorb` | [D3PM](https://arxiv.org/abs/2107.03006): absorbing variant, discrete-time variational objective plus reconstruction. |
| `sedd_absorb` | [SEDD](https://arxiv.org/abs/2310.16834): absorbing scores, score entropy and analytic sampling. |
| `udlm` | [UDLM](https://arxiv.org/abs/2412.10193): existing plug-in `uniform_diffusion` recipe. No paper guidance extensions. |
| `uniform_posterior` | Uniform corruption with ordinary posterior CE; distinct from plug-in UDLM. |
| `duo` | [DUO](https://arxiv.org/abs/2506.10892): continuous-relaxation/discrete curriculum, uniform reverse inference. |
| `dfm_masked`, `dfm_uniform` | [Discrete Flow Matching](https://arxiv.org/abs/2407.15595): posterior CE and base CTMC probability-velocity updates. No extra corrector. |
| `gidd_elbo` | [GIDD](https://proceedings.mlr.press/v267/von-rutte25a.html): mask/uniform mixture with composed ELBO. `gidd_corruption` is the CE baseline. Neither is dynamic GIDD+. |
| `candi_hybrid` | CANDI masked/Gaussian hybrid with mask-dependent fusion and rank calibration. |
| `cadd` | [CADD-style](https://arxiv.org/abs/2510.01329) gated Gaussian/masked hybrid. YAML records missing dropout, sharing/detachment and schedule/sampler alignment. |
| `cdcd` | [CDCD-style](https://arxiv.org/abs/2211.15089): learned embedding Gaussian CE and adaptive clock; adaptation machinery differs from the paper. |
| `flm_gaussian` | [FLM-style](https://arxiv.org/abs/2602.16813): Gaussian one-hot linear flow and error-based clock; finite calibration bounds and local conditioning differ. |
| `langflow` | [LangFlow-style](https://arxiv.org/abs/2604.11748): embedding Gaussian CE, Gumbel clock, expected-embedding SC. Normalization, conditioning and architectural details still differ. |
| `simplex` | [Simplex Diffusion Models](https://arxiv.org/abs/2609.35553): Dirichlet bridge construction, CE and churn. Not Dirichlet Flow Matching or SSD-LM. |
| `spherical_flow_matching` | [S-FLM](https://arxiv.org/abs/2605.11125) geodesic sphere path and tangent integration. Not every paper architecture/velocity approximation. |
| `spherical_vmf` | [vMF spherical flows](https://arxiv.org/abs/2605.05629), with predictor-corrector inference. Paper classifier/damping details differ. |
| `replaid` | [RePlaid](https://arxiv.org/abs/2605.18530): learned embeddings and schedule, reconstruction/diffusion/prior terms. No separate original-PLAID preset. |
| `blurring_diffusion` | Token-CE language adaptation of [Blurring Diffusion](https://arxiv.org/abs/2209.05557). |
| `llada_style` | [LLaDA-style](https://arxiv.org/abs/2502.09992) masked training with local DiT and ancestral decoding. |
| `ardm_style` | Basic absorbing order-agnostic [ARDM-style](https://arxiv.org/abs/2110.02037) construction, without upscaling/adaptive-budget machinery. |

Related names deliberately share defaults instead of duplicating implementation
or pretending to be distinct algorithms. A recipe may override a sampler, loss,
representation or schedule using the same Hydra groups as other experiments.

## Modifiers and inference recipes

| Experiment | Construction and limits |
| --- | --- |
| `blockgen_masked`, `blockgen_uniform` | [BlockGen fixed-block baselines](../methods/blockgen.md): CE masking or plug-in uniform ELBO; no block-size mixture or ARPC. |
| `bd3lm` | [Block diffusion](https://arxiv.org/abs/2503.09573) with clean prefix, configurable `block_diffusion.block_size`, and ancestral sampling. No claim to the optimized paper schedules. |
| `eb_masked` | [Entropy-bounded unmasking](https://arxiv.org/abs/2505.24857) on MDLM. This changes inference, not training. |
| `flexmdm_style` | [FlexMDM-inspired](https://arxiv.org/abs/2509.01025) masked insertion. Uses our count/gap objectives, not the full paper variational objective. |

Self-conditioning is still an independent override, for example
`self_conditioning.enabled=true self_conditioning.source=posterior_mean` on a
compatible recipe. Insertion is still the `variable_length` modifier; forward
deletion does not mean the reverse sampler supports deletion/substitution edits.
Set task-specific variable-length evaluation options when using insertion.

No named config is provided for full SemiCat/DoubleCat, GenMD4, Edit Flows, DID,
ReMDM or Dirichlet Flow Matching: the missing components need code. In particular,
`semicat_gaussian` remains a Gaussian baseline, not full SemiCat.
