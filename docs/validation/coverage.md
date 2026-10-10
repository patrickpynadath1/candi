# Validation coverage

Implementation, successful execution, and reproduction are different claims.
**Implemented** means a maintained code/config path exists. **Execution tested**
means that a specified fixture or integration ran. **Reproduced** means a
comparison against a defined reference succeeded. A short pilot is evidence
that a method can learn, not proof of convergence or paper-level performance.

The regression suite with Ray Tune sweeps passed **476 tests**, including
available CUDA checks (417 before adding sweeps).
That total is a snapshot, not a promise that every method,
task, precision, sampler, modifier and distributed strategy combination has
been tested. Run the current suite from the repository root:

```bash
uv run python -m pytest core/tests -q
```

GPU and real sandbox checks require those facilities; skipped tests do not
establish their behavior on the current machine.

## Model and sampler composition audit

The shared `TextDiT` and fixed-grid sampler refactors were developed on separate
branches from `babad72`, then merged and compared against that baseline. A CPU
audit instantiated 16 small method presets with fixed seeds and dropout disabled,
performed two AdamW updates, and generated prompted sequences. Losses, final
gradients, updated state dictionaries, generated tokens and sampling metadata
matched exactly after both refactors were merged.

The audit covered MDLM, uniform plug-in and posterior diffusion, SEDD, DUO's
discrete phase, D3PM, one-hot and embedding Gaussian diffusion, spherical flow,
simplex, blurring, RePlaid, autoregressive and permuted autoregressive models,
CANDI and uniform/simplex hybrids. This is refactor equivalence on small fixtures,
not a new training-quality or convergence result.

Permanent regression tests additionally compare discrete sampling against frozen
pre-refactor loops, including intermediate prediction inputs and RNG state, and
compare shared model execution and gradients with the original direct block/head
calculation. Existing modifier, mixed-precision, cache and checkpoint tests remain
part of the complete suite.

## Programmatic sweep execution

All 45 experiment presets resolve through the Hydra/Ray Tune adapter without
launching training. The sweep API, scheduling and experiment state now belong
to Ray Tune; the previous custom executor has been removed.

Small MDLM and autoregressive trials trained on distinct A6000 GPUs through
Ray Tune for two steps using BF16 and synthetic data. Both saved durable Tune
checkpoints containing the runner's Lightning checkpoints and resolved config.
Separate evaluation trials loaded these checkpoints and generated four records
each. An additional check evaluated the same MDLM checkpoint with ancestral and
entropy-bounded samplers as independent inference trials. Restoring the training
sweep did not rewrite completed checkpoints.
This verifies execution and restoration, not generation quality.

The dynamic job queue additionally passed a two-GPU interruption/resume check
using the dataclass job definitions: an MDLM training job produced ancestral and
entropy-bounded inference while an autoregressive training job remained open.
After an intentional SIGINT, `Tuner.restore` finished five unique jobs without
rerunning the completed training or inference jobs. Queue unit tests cover
pending/issued state, duplicate completion notifications, inference priority,
failed parents, and independent configuration snapshots.

The [base algorithm sweep](../guides/base-algorithm-sweep.md) has 11 additional
catalogue tests. The reduced single-seed plan composes 66 supported training configurations;
all instantiate with small model dimensions, and all 360 associated inference
sampler configurations instantiate. Preview/no-launch and unsupported-combination
checks pass. This is construction coverage, not training or quality validation of
the full catalogue. Uniform plug-in self-conditioning remains explicitly blocked.

Regression tests cover all preset configs, independent editable configs, nested
Hydra overrides, unsupported resource/config combinations, aggregate-only
metrics, real CPU Tune training, failure reporting, durable checkpoint artifacts,
and restoration. The adapter supports CPU or one-GPU trials and reports final
metrics; it does not implement intermediate reporting, automatic partial-training
resume or distributed training within a trial. See the
[programmatic sweep guide](../guides/programmatic-sweeps.md) for examples and limits.

## Methods and tasks

| Area | Evidence | Boundary of the claim |
| --- | --- | --- |
| MDLM / Sudoku | Exact saved-checkpoint generation parity for ancestral and EB over 2,000 puzzles; controlled loss and gradients identical | No fresh 20,000-step convergence reproduction |
| Gaussian one-hot | 2,000-step Sudoku pilot; Euler/Heun/ODE evaluations; exact before/after representation-extraction regression | Sampler intentionally differs from original continuous implementation |
| Gaussian embeddings | Three-step CUDA/BF16 train/checkpoint smoke; all three continuous solvers; sphere-rank reference equality | No trained embedding-model quality benchmark |
| Uniform, posterior-uniform, SEDD, DUO, D3PM | Process/objective/config/sampling regression tests | No full-budget pretrained quality reproduction claimed |
| Autoregressive / permuted AR | Causal visibility, KV-cache equivalence and execution tests; permuted-AR 10k-step Text8 pilot | Ordinary AR quality not established; Text8 pilot is a small model |
| Spherical flow matching | CUDA geometry/adaptation tests and 2k-step Sudoku run | Early run remains inside 2,500-step warmup, not 20k-step reproduction |
| Simplex | Copied numerical primitives and nine bridge combinations match source exactly; CUDA training/sampling cases | No standalone trained Sudoku quality claim; source paper protocol differs |
| RePlaid | Corrected reference output/loss/gradient audits; training and sampling checks; corrected 2k-step pilot | Successful reference used 50k steps and different data; not reproduced here |
| Blurring | Process/schedule/model/sampling and block-boundary tests | No full-budget quality reproduction recorded |
| CANDI hybrid | Close imported-checkpoint result plus fresh standard-mask 2k-step run | Standard masking intentionally changes training corruption |
| Uniform + simplex hybrid | Full 2k-step training and 2,000-puzzle evaluation | Single pilot, no tuning or convergence claim |
| Sudoku task | Full checkpoint parity; answer sanitization, metric/dataclass parity | Puzzle prompt preservation and answer clue consistency are separate |
| Text8 task | All 35 token IDs and 25 controlled packed windows match original; real data smoke and generation metrics | Multi-worker legacy packing boundaries are not reproduced generally |
| OpenWebText task | Controlled metric math parity, training fixture, cached scorer and real pretrained MDLM point | BOS boundary bug intentionally fixed; no full training reproduction |
| TinyGSM task | Fixture token/mask parity, two-step training/evaluation, real Bubblewrap checks | Finite data only; no full training or accuracy reproduction |

See [measured results](reproduction-results.md) for exact counts, budgets and
weight selection. Reusable components and method settings are documented in
[architecture](../concepts/architecture.md) and [methods](../index.md).

## Modifiers and infrastructure

Self-conditioning covers posterior-mean and hidden sources, detached bootstrap,
zero-initialized feedback, checkpoint behavior, and sampler sessions. The MDLM
hidden-feedback pilot is recorded separately from earlier superseded feedback
designs. Adaptive ODE sampling rejects self-conditioning because rejected-stage
feedback rollback is not implemented. Generic posterior-mean feedback requires
probability outputs; that is checked at construction/use boundaries.

Block diffusion has attention, masks, loss-routing, prefix-cache and combination
tests. It does not imply every sampler or model can consume the modifier.
Consult [block diffusion](../modifiers/block-diffusion.md) for supported paths.
Block-local forward deletion/reverse insertion now has training, generation,
future-token isolation, padding, protected-prefix and per-block capacity tests.
Approximately 22M masked and Gaussian Text8 models with self-conditioning passed
GPU smoke tests and produced trained block samples. Gaussian, masked, simplex
and spherical whole-sequence insertion runs also executed training and sampling.
These demonstrate composition and trainability, not converged language quality.
See the [recorded insertion pilot evidence](../modifiers/variable-length.md#what-has-been-demonstrated)
for a checkpoint-specific snapshot and its limitations.

| Infrastructure | Executed check | Remaining qualification |
| --- | --- | --- |
| DDP | Two GPUs, four MDLM steps, cosine scheduler, EMA, callback generation, checkpoint | Not a matrix of all algorithms/modifiers |
| Restricted FSDP | Two GPUs, two steps, resume to four, standalone checkpoint evaluation | EMA, callback generation, norm clipping and parameter-projection hooks unsupported |
| Native resume | Optimizer, scheduler, EMA and RNG restoration | Exact mid-epoch shuffled/prefetched sample replay not guaranteed |
| LR schedules | Constant, constant-warmup, halving and cosine tests | Cosine is local implementation; missing legacy class not a parity reference |
| Frontier sweeps | Cartesian-grid/report/Pareto/resume tests and one pretrained OpenWebText point | No completed multi-point real-model frontier claimed |
| Self-containment | Isolated-tree entrypoint/config and targeted tests, then root runner/config relocation | External Python dependencies, datasets, weights and scorers still required |

## Explicit omissions

Maze and Zebra tasks, legacy hub-path workflows, latent flow matching,
directly predicted continuous velocity, RePlaid hybrids, and simplex teacher/
student distillation are not included. Learned contextual or compressing
encoders are not implemented by the tokenwise representation interface.
Legacy optimizer-state import and bitwise interrupted-run data replay are also
outside the current guarantees. Removing legacy source did not turn these
omissions into implemented functionality.
