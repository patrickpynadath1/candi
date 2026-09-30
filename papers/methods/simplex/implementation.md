# Simplex Diffusion Models

- Status: `implemented` (`triage` → `scoped` → `implementing` → `implemented` → `reproduced`)
- Slug: `simplex`
- Paper ID: arXiv:2609.35553
- Source: [2609.35553.pdf](../../inbox/2609.35553.pdf)
- Source SHA-256: `b6d5fa2313cbbb9d0c4f2838cb62fcdba9e2f31e32b1ebb034a5e5ee1108b9f9`
- Added: 2026-09-30
- Owners: unassigned

## Claim and scope

Implement arXiv:2609.35553 in Duo: simplex corruption and reverse bridges,
concentration schedules, expectation/argmax inputs, reusable latent self-conditioning,
adaptive time sampling, guidance and distillation, and the Sudoku reproduction.
Implementation and empirical reproduction are separate milestones.

## Paper specification and source-to-code map

| Component | Paper reference (PDF page) | Specification / code |
| --- | --- | --- |
| Corruption | Algorithm 1, p23 | beta=c(t)((1-t) onehot(x)+t pi); sample Dir(beta); `simplex_utils.corruption_parameters` |
| Objective | Eq11, p6 | Unweighted target-token cross entropy; `SimplexDiffusion.nll` |
| Concentration | Section4.2, p7 | nu constant or constant then linear; c=1/nu-1; also c=epsilon/t |
| Reverse bridge | Prop3.3, p4, Algorithm2 p23 | rho=(1-kappa)min(1,cs*s/(ct*t)); Beta thinning, Dirichlet innovation, independent Beta mixing; `reverse_bridge` |
| Endpoints | Appendix A p21, C.1 p24 | Face distributions interpreted exactly; kappa=1 fresh corruption; final clean vertex sampled from denoiser |
| Gamma | Algorithm3 p25 | Marsaglia-Tsang rejection and alpha<1 boosting in log space; `log_gamma_sample` |
| Inputs | J.2 pp47–48 | Expected embedding P@E or argmax lookup; full simplex retained by sampler |
| SC | I.3 pp44–45, Eq27 | Last hidden latent added through zero-initialized affine LayerNorm; detached first pass with p=.9; previous latent at inference |
| Time sampler | J.1 p47 | 50 bin mean loss, linear interpolation, positive derivative, density EMA .9, 1000-point inverse CDF; warmup1000/refit50 |
| Guidance | C.2 pp24–25 | Log probability interpolation (1-w)logP_u+w logP_c |
| Distillation | G.2 pp38–39, Algorithm4 | Reparameterized student corruption; stopped auxiliary-minus-teacher predictions; generator surrogate a_t sum(log(Pt)*difference) |
| Sudoku | J.4 p50 | 200k train/5k validation, unique solutions, 30 clues, no shared full grids, 180 tokens / vocabulary14 |
| Training | J.4 p50 | 8 blocks, hidden512, heads8, MLP4x, RoPE, untied embeddings, dropout.1, Adam3e-4, warmup2500, no weight decay, clip1, EMA.9999, fp32, batch256, 50k steps / SC38462 |
| Evaluation | Table1 p7, M.1 pp76–79 | Exact match, 180 sampling steps, five seeds, linear/cosine/adaptive grids, churn0/.2/1 |

## Decisions and ambiguities

- The paper does not specify the ring-buffer capacity or empty-bin policy. Use
  65536 examples and interpolate observed bin centers (flat beyond observed centers).
  Refit only once per optimizer step; gather observations across distributed ranks.
- Use the repository's backtracking generator (same strategy described in J.4),
  reject puzzles that cannot reach exactly30 clues. Exact original dataset/seed list
  is not provided. Record generated dataset seed and hashes.
- Gamma draws stay in log space to avoid small-shape underflow, including thinning.
  Rejection decisions are nondifferentiable; accepted proposals retain pathwise
  derivatives, as described by the paper's Algorithm3 distillation implementation.
- Global SC is opt-in. Existing RePlaid embedding SC is distinct; disable that
  feature when comparing the paper's latent SC alone.
- KV-cached autoregressive samplers fall back to uncached sampling when SC is
  enabled. Growing causal prefixes pad the previous latent. External Hugging Face
  backbones without a transformer block stack are rejected explicitly.

## Acceptance criteria

- Dirichlet corruption and oracle reverse-bridge means/variances agree with theory.
- Churn endpoints, simplex faces, tiny concentrations, guidance identities work.
- Tiny real models have finite loss and finite nonzero gradients; prompts are clean.
- SC gradients do not traverse first pass or previous sampling steps; batches reset.
- Hydra registrations, sampler contracts, checkpoint reload, full-float32 path pass.
- Every training/evaluation/smoke experiment records config, seed and metrics in W&B.
- Five-seed full-budget validation metrics are compared with Tables1/21–24.

## Reproduction status

Full-budget reproduction is running; no paper-accuracy claim is established yet.
The workspace summarizer reports five-seed evaluation coverage separately from
accuracy agreement, which still needs comparison against the paper. The frozen
queue's older summary field `complete_five_seed_reproduction` measures coverage
only; it must not be interpreted as evidence that paper accuracy was matched.

The audited dataset reuses the existing `sudoku_hard_train200000_valid10000_seed42`
cache: all 200k training puzzles and the first 5k validation puzzles. All 205k
puzzles were independently checked for exactly 30 clues, Sudoku validity, clue
agreement, one solution, and no repeated full grids. Token SHA-256:
`67c021bebcbd1c71fc87344585a54af06d237e1cc51599da42e241152819b219`.
The paper does not publish the exact generated grids or seed list; this is a
protocol reproduction, not a claim of identical original examples.

The queue covers SDM Tables21–23 and the masked/uniform rows of Table24:
160 training runs and 1050 standalone evaluations, seeds1–5, all five concentration
schedules, uniform/adaptive training times, SC on/off, three inference grids where
available, churn0/.2/1, and ancestral/PC discrete baselines. Existing AR/FLM/S-FLM
methods accept the new SC option, but their Table24 reference experiments and the
separate DFM comparison are not included in this queue. Reproduction of every row
of the paper's Sudoku tables therefore remains unfinished.

- W&B project: https://wandb.ai/rz-text-dlp/simplex-sudoku
- Queue PID at launch: 382750; GPUs1,2,3,5.
- Artifacts: `/data/shared_data/patrick-duo/simplex-paper-20260930-0303`.
- Frozen runtime source: `source-v4-current/` under that directory.
- Snapshot SHA-256: `1334e9cb33490e3ae1a4c719b75f5b74a64210a5437571d8b721ca5d46794936`.
- Current first-wave runs: SDM+SC `ufm0i0g1`, SDM `ncctygqd`, MDLM-CE+SC
  `swftyi2b`, UDM+SC `iba4k3jm`. Each resumed model, optimizer, and scheduler
  state from its latest saved checkpoint at step10000.
- At the user's request, the running queue now uses `--weights current`:
  `eval.disable_ema=true` for training-time validation/sampling and standalone
  evaluation. TrainerBase previously ignored this flag during training-time
  evaluation; both weight swapping and restoration now honor it. EMA continues
  to be tracked in checkpoints but is never selected for this queue's evaluation.
  This is an explicit departure from the paper's EMA evaluation protocol.
- Original checkpoints are preserved as `restart_from_ema_v3_step10000.ckpt`;
  previous logs/configs/samples are under each run's `history/ema-v3/`. The old
  queue manifest and launch metadata are under the root `history/ema-v3/`.
  New W&B runs are tagged `weights-current` and named with `_current`.
- Restart validation:61 tests passed; a W&B-logged resume smoke test
  (`qwhl4jbl`) restored step800, advanced to802, and sampled current weights.
- `manifest.json` contains every exact command; `queue_process.json` records the
  launcher; per-run `queue_status.json` and `process_*.log` record progress/failure.
- Workers retain rolling and final checkpoints, stop on subprocess failure, and
  refuse another phase with less than20GiB free storage. Evaluation loads the
  final-step checkpoint, including for SC runs ending at38462 rather than a
  periodic checkpoint at30000.

## Additional implementation decisions

- Workspace maintenance: experiment entry points live together under
  `scripts/simplex/` (see its README). The external frozen runtime retains its
  original paths; no running jobs are migrated by this source cleanup.
- The paper time sampler is attached only to diffusion models that request it
  through adaptive sampling, the paper protocol, or simplex sampling. Other
  algorithms do not allocate it. Existing adaptive checkpoint keys are retained.
- Final checkpoint writing is controlled by `checkpointing.save_final`, enabled
  by the Sudoku experiment and independent of `require_wandb`. W&B lifecycle
  handling now includes errors during dataset and model setup.
- Masked and uniform paper samplers use the declared `discrete_reverse_probs`
  capability (`reverse_transition_probs`) instead of a private DUO method.
- Sudoku prompt extraction belongs to the tokenizer, covering both serialized
  layouts. Self-conditioning installation rejects accidental double wrapping;
  its adapter design and parameter paths remain unchanged.

- The final paper configuration uses fixed EMA decay .9999, with the repository's
  update-count warmup disabled. Time-independent paper baselines receive an
  actual zero AdaLN conditioning vector. The existing attention/MLP output
  dropout is .1; extra dropout on the vocabulary head is disabled.
- Gamma/Dirichlet state arithmetic uses float64 log probabilities to avoid
  small-shape underflow; neural training uses float32.
- Conditional-dropout training (zero prompt embeddings) enables optional CFG.
  Its scale and the J.8 confidence sampler are disabled for the Sudoku matrix.
- `simplex_dmd` implements Algorithm4 with a deterministic generator, independent
  time draws from the configured time sampler, an auxiliary denoiser conditioned
  on both source and target times, and a frozen checkpoint teacher. This is an
  explicit choice of the generic G and Q in Algorithm4; the paper does not give
  sufficient details to assert bit-identical TinyGSM distillation. Distillation
  has only been smoke-tested, not reproduced at scale.
- Validation intervals span epoch boundaries (`check_val_every_n_epoch=null`).
  Training drops the last incomplete accumulation group, so every optimizer step
  has256 examples. The first full-queue startup caught the original invalid
  within-epoch validation interval before training; failed runs and local
  `startup_failure*.log` files remain available.
- Optional global latent SC is separate from RePlaid's embedding SC. Set its
  existing `algo.self_condition_rate=0` when measuring latent SC alone.
- SC forward/backward tests cover the shipping algorithm-config matrix (including
  AR, block diffusion, spherical/continuous methods and RePlaid). Sampling tests
  cover SDM and masked/uniform ancestral/PC paths, prompt preservation, detached
  carry, and reset on failure. Other specialized sampling combinations are not
  claimed to have been empirically validated.

## Commands and results

Workspace cleanup validation: **370 tests passed** across algorithm construction,
loss/backward, sampler contracts, self-conditioning, checkpoint/EMA lifecycle,
Sudoku layouts, dataloading, and the reproduction manifest. New-code Ruff checks,
script CLI checks, shell syntax, and `git diff --check` passed. Repository-wide
lint still includes unrelated pre-existing issues.

Generated Python/test/lint caches were removed. Completed top-level W&B logs
were preserved under `runs/simplex-worktree-logs/wandb/`; checkpoints, datasets,
results, and the external frozen runtime were retained.

### Sampler audit after near-random early samples

The full reverse loop was checked against Algorithm2, including prior,
categorical clean-vertex draws, Beta thinning/mixing, churn boundaries, and the
final categorical endpoint. A new exact-Bayes-denoiser test recovers a known
non-uniform data distribution through the entire chain at churn0/.2/1. The SDM
test suite passed23 tests after this addition.

An independent800-step, two-layer diagnostic compared identical sampling seeds,
180 NFEs, and32 held-out puzzles using current versus EMA weights. Current
weights obtained52.0/53.4/53.1% reported cell accuracy at churn0/.2/1; EMA obtained
6.7/7.5/7.6%. Both had0% complete solves. This supports initialization-heavy EMA
as a cause of random early sampling, without establishing final reproduction
accuracy or directly measuring the running28.6M models' current weights.

The fixed EMA specified by Appendix J.4 retains an initialization coefficient
`0.9999 ** step`:92.3% at800 steps and60.7% at5000 steps. This is a parameter
averaging coefficient, not a token probability. Current-weight diagnostic
evaluation can use `eval.disable_ema=true`; official paper evaluation retains
EMA. No sampler equations or ongoing training jobs were changed by this audit.

- Training: https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/28v31ua3
- Comparison: https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/bn7y6z35
- Reusable diagnostic: `scripts/simplex/audit_sampler.py`, with required
  `--config`, `--checkpoint`, and `--output` arguments. It logs metrics, synthetic
  Sudoku examples, and a comparison artifact to W&B.

Validation used `/home/patrick/duo/.venv/bin/python` with `OMP_NUM_THREADS=2`.

- Consolidated suite: **316 passed** across real configs, diffusion, SDM, SC,
  Sudoku, callbacks, dataloader, sampler contracts and DiT.
- Reproduction-manifest test: **1 passed**, checking all160/1050 jobs and composing
  representative training/evaluation commands with Hydra.
- Final focused suites after EMA/time-conditioning refinements: **62 passed** and
  **39 passed**. A final SC suite including growing-prefix AR sampling passed
  **38 tests**. New-code Ruff checks and `git diff --check` passed.
- The repository-wide Ruff check still reports15 pre-existing issues (unused
  imports and type-comparison style); these are outside this implementation.
- Six two-step CPU training runs (SDM/MDLM/UDM, SC off/on) succeeded and synced to
  W&B. A paper-size28.6M float32 GPU pilot ran50 optimizer steps, and separate
  tiny DMD-training and standalone evaluation runs succeeded. Pilot/smoke results
  are execution checks and are not estimates of final accuracy.

W&B execution evidence:

| Check | Run |
| --- | --- |
| SDM without SC | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/zsamjm4r |
| SDM with SC | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/3xmalhe2 |
| MDLM without SC | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/k2aewyt5 |
| MDLM with SC | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/xay6hat7 |
| UDM without SC | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/bz9bsyxd |
| UDM with SC | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/hdb4noji |
| Dataset audit | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/zm1qw9t8 |
| GPU pilot | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/wps72rta |
| Distillation smoke | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/l5513e0m |
| Standalone evaluation | https://wandb.ai/rz-text-dlp/simplex-sudoku/runs/z19sfazd |

The initial CPU attempt failed in the existing loader's visible-device counting
before training; it is also retained in W&B (`ulq8l7ly`). The loader now respects
the requested device count, including CPU and single-GPU runs on an eight-GPU host.

The four initial GPU attempts also failed before training because Lightning's
validation interval (5000 batches) exceeded the epoch length (3125 batches).
`check_val_every_n_epoch: null` enables cross-epoch validation; the configured
training-batch limit also keeps gradient-accumulation groups complete. These
failures are retained and tagged `superseded-by-fixed-run` in W&B, with recovery
notes linking each replacement:

| Failed attempt | Healthy replacement |
| --- | --- |
| `3y1cxakr` | `sabrfg79` (SDM+SC) |
| `o9qit7p8` | `ub0kmuvq` (SDM) |
| `my58cye4` | `l0s18bc4` (MDLM+SC) |
| `8zp4c37d` | `61vkpxft` (UDM+SC) |

A subsequent failure audit verified all four replacements running beyond5000
optimizer steps, with finite training losses and validation NLLs. No additional
failed runs were present in the project. Reproduction-config and dataloader
regression checks passed24 tests. Healthy jobs were left running without restart.

```bash
# Review the complete matrix without launching training.
python scripts/simplex/reproduce.py --full --phase plan
# Opt-in latent SC for an existing method.
python main.py +experiment=sudoku_simplex_mdlm self_conditioning.enabled=true trainer.max_steps=38462
# Inspect the running frozen reproduction, without starting a second queue.
cat /data/shared_data/patrick-duo/simplex-paper-20260930-0303/queue_process.json
# Aggregate only completed full-validation results.
python scripts/simplex/reproduce.py --phase summarize --output /data/shared_data/patrick-duo/simplex-paper-20260930-0303
```
