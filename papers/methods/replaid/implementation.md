# Continuous Diffusion Scales Competitively with Discrete Diffusion for Language

- Status: `implemented` (`triage` → `scoped` → `implementing` → `implemented` → `reproduced`)
- Slug: `replaid`
- Paper ID: arXiv:2605.18530v1
- Source: [2605.18530v1.pdf](../../inbox/2605.18530v1.pdf)
- Source SHA-256: `e71b53f3a2edcc94f302f5298f07bf31f2f8e0e7360b208304bb989450927db0`
- Added: 2026-08-16
- Owners: unassigned

## Claim and scope

### One-sentence claim

RePlaid keeps Plaid's likelihood-derived continuous embedding diffusion while
aligning its transformer and optimization setup with modern discrete DLMs,
closing much of the previously reported scaling gap.

### In scope

- The unit-normalized, learnable low-dimensional token embedding (`d_e=16`).
- VP Gaussian corruption, categorical clean-token prediction, output-prior
  logits, and the complete NELBO objective.
- Learnable monotone noise schedule and its variance-minimizing gradient hook.
- Optional self-conditioning and a DDPM correctness path before faster solvers.
- Architecture/config alignment that is required for a fair Duo/MDLM comparison.

### Out of scope

- Paper-scale training and the claim of 22.1 OWT perplexity in the first change.
- ODE likelihood evaluation, DPM++/Heun solvers, and scaling-law fitting until the
  DDPM implementation passes the small experiment ladder.

### Dependencies and closest existing implementation

- Paper/code dependencies: Plaid/VDM and the modern DiT setup cited by the paper;
  check whether official RePlaid code/configs are available before implementation.
- Closest Duo algorithm: `embedding_diffusion` / `algos/cont_diff_embed.py` for
  continuous embeddings; `mdlm` and `duo` for architecture-aligned baselines.
- Reusable components: schedule machinery in `algos/noise_schedules.py`, the model
  registry, `TrainerBase`, standalone samplers, and Hydra experiment configs.

## Paper specification

For every implementation-relevant item, cite a section, equation/algorithm, and
PDF page. Use symbols from the paper and state tensor shapes and conventions.

| Component | Paper reference | Executable specification |
| --- | --- | --- |
| Data representation | Sec. 2, p. 2 | Map one-hot `x:[B,L,V]` through learned `E:[V,d_e]`; constrain each row of `E` to unit norm; default `d_e=16`. |
| Forward/corruption process | Eq. 1–2, p. 2 | `q(z_t|x)=N(alpha_t xE, sigma_t^2 I)`, VP constraint `alpha_t^2+sigma_t^2=1`, and the stated Gaussian posterior for `s<t`. |
| Model prediction/parameterization | Eq. 3, p. 2–3 | Denoiser predicts categorical `x_theta:[B,L,V]`; clean embedding is `x_theta E`; add the closed-form Gaussian output-prior logits before softmax. |
| Training objective | Eq. 4 and Alg. 1, pp. 3, 20–21 | Sum prior KL, Monte Carlo reconstruction CE at `t=0`, and `-0.5 SNR'(t)||e_theta-e||^2`; preserve the paper's batch allocation and reductions. |
| Noise/time schedule | Sec. 2, p. 3; Fig. 8, p. 37 | `gamma=gamma_0+(gamma_1-gamma_0) gamma_tilde(t)`, monotone `gamma_tilde`; learn endpoints by loss and interior by the variance-minimizing hook. |
| Reverse process/sampler | Eq. 2–3, p. 2; Sec. 4.1 | Plug predicted categorical clean state into the Gaussian posterior; implement/validate uniform-time DDPM first. |
| Architecture changes | Sec. 3.1, p. 3; Tables 5–6, pp. 38–39 | Bidirectional DiT with RoPE/AdaLN-Zero, LayerNorm, biased MLP, GELU(tanh), gated residuals; do not retain Plaid's FP32-only final head. |

## Source-to-code map

| Paper item | Planned code | Verification |
| --- | --- | --- |
| Eq. 1–4 and Algorithm 1 | `algos/replaid.py` | corruption moments, NELBO terms/reductions, finite backward |
| Low-dimensional embedding and self-conditioning inputs | `models/replaid_dit.py` (or a minimal extension of an existing DiT) | normalization, shape, detach/bootstrap, and gradient tests |
| DDPM reverse process | `samplers/replaid_ddpm.py` | posterior mean/variance limiting cases and capability contract |
| Paper defaults | `configs/algo/replaid.yaml` | Hydra registry and tiny-real-config tests |
| First controlled comparison | `configs/experiment/replaid_<task>.yaml` | fixed-token-budget baseline comparison |

## Acceptance criteria

### Mathematical/unit

- [x] VP identity, monotone schedule, and reverse transition limiting cases are
  covered; empirical moment testing remains useful at larger sample count.
- [x] Reverse posterior
  reduces correctly at adjacent/end times.
- [ ] Each NELBO component matches a hand-computed tiny tensor, including padding
  masks and sequence-vs-token reductions.
- [x] Unit embedding rows are projected after optimizer steps and receive
  finite non-zero likelihood gradients.
- [x] Schedule derivatives, SNR weighting, and output-prior logits remain finite
  near configured endpoints.
- [x] Self-conditioning bootstrap is detached and is used for the configured
  training fraction and every sampling step.

### Integration

- [x] Shipping config resolves registered algorithm and backbone
- [x] Tiny real model returns finite loss and non-zero finite gradients
- [x] Sampler validates the model capability contract
- [ ] Checkpoint save/load works if state structure changes

### Reproduction

- [ ] Dataset/tokenization and preprocessing match the paper
- [ ] Training budget and evaluation metric are recorded
- [ ] Baseline and target result include uncertainty/tolerance
- [ ] Exact command, config, seed, commit, and artifact path are recorded

## Experiment ladder

1. Analytic/unit test: verify Eq. 1–4, posterior coefficients, schedule
   monotonicity, embedding projection, and self-conditioning detach semantics.
2. Tiny overfit or CPU smoke test: overfit a fixed text8 batch; require decreasing
   NELBO and successful 4–8 step DDPM generation.
3. Small comparison against closest baseline: fixed model/data/token budget
   against `embedding_diffusion` and MDLM; report NELBO and sample metrics.
4. Paper-scale reproduction: reproduce one OWT configuration from Table 9 before
   attempting the full scaling frontier; compare the same bound and tokenizer.

## Deviations from the paper

| Deviation | Reason | Expected effect | How tested |
| --- | --- | --- | --- |
| Positive-weight monotone basis instead of an unreleased monotone-MLP implementation | The paper specifies behavior but no source code is linked | Same monotonicity/endpoints, potentially different optimization dynamics | monotonicity, derivative, VP, and backward tests |
| Component LR groups use the repository scheduler rather than a paper-scale muP setup | Keep the first implementation runnable in Duo | Small-scale behavior is testable; paper-scale parity is not claimed | real-config backward test; reproduction remains unchecked |
| Only DDPM is implemented | It is the paper's default correctness path | No DDIM/DPM++/Heun speed comparison yet | Appendix-D zero-gap and finite reverse-step tests |
| Conditional generation uses forward-consistent inpainting | The paper reports unconditional generation and does not specify prompting | Enables TinyGSM/GSM8K evaluation without altering unconditional DDPM | fixed-noise prompt-trajectory unit test and real 56.2M GPU prompt smoke test |

## Open questions and decisions

- [x] Checked the authors' project page on 2026-08-16; it links the paper but no
  official implementation. Re-check before reproduction work.
- [x] A dedicated `replaid_dit` is used because the existing continuous DiT has a
  different VE/preconditioning contract.
- [x] Implemented Gaussian output-prior logits and Appendix-B adaptive
  reconstruction/diffusion allocation.
- [x] Implemented separate embedding/schedule/endpoint optimizer groups and
  float64 schedule calculations; paper-scale muP is intentionally deferred.
- [ ] Define which dataset is the inexpensive acceptance target (text8, PTB, or a
  small OWT slice) and the quantitative tolerance against the closest baseline.

## Commands and results

```bash
uv run pytest tests/unit/algos/test_replaid.py \
  tests/unit/samplers/test_replaid_ddpm.py \
  tests/unit/algos/test_registry.py \
  tests/unit/algos/test_real_configs.py \
  tests/unit/samplers/test_registry.py -q
```

| Date | Commit | Config/seed | Result or artifact | Interpretation |
| --- | --- | --- | --- | --- |
| 2026-08-16 | working tree | tiny text8 tokenizer, seeds 1–3 | 91 focused/integration tests passed | Training, backward, registry, and 2-step DDPM generation paths work; no paper-scale claim |
