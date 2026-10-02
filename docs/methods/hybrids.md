# Parallel-view hybrids

A hybrid maintains multiple noisy views of the **same clean tokens**. Each branch has a process, schedule, local map from global time, and input adapter. The model sums branch embeddings and evaluates one shared transformer and clean-token posterior. There is no required conversion from one branch's state space into another's.

At a joint sampling step, branches retain and update their own states using that shared posterior, then fuse their embeddings for the next prediction. Branch updates and adapters still have a cost; the expensive transformer is shared rather than duplicated.

## Existing assemblies

```bash
uv run python main.py +experiment=candi_hybrid
uv run python main.py +experiment=uniform_simplex_hybrid
```

| Preset | Branches | Sampler | Special policy |
|---|---|---|---|
| `candi_hybrid` | Absorbing masking + rank-calibrated one-hot Gaussian | Hybrid entropy bounded | Synchronize committed tokens |
| `uniform_simplex_hybrid` | Posterior uniform + simplex | Fixed-step hybrid | Independent states |

CANDI retains the reference input fusion and frozen-posterior Gaussian update. Its explicit `synchronize_committed=true` policy sets the continuous view of committed mask tokens to the same clean token before the next prediction. Other presets leave this off. The masking branch also fixes revealed-token posterior outputs.

Training masking intentionally corrects a historical bug: selected positions always become masks. The old path redrew uniform tokens and inferred revelation from equality with the clean token, accidentally retaining some selected positions. Thus importing an old checkpoint can validate inference, but cannot establish equal training distributions.

## Composition contracts

Every branch must consume a clean-token posterior. Uniform branches therefore use `PosteriorUniformProcess`, not the plug-in uniform transition. The framework rejects plug-in branches; see [discrete diffusion](discrete-diffusion.md) for why the mathematics differs.

The current objective is token CE. RePlaid hybrid likelihood objectives are not implemented. Likewise, an arbitrary process is not automatically compatible: it needs a branch adapter that defines corruption, prior, embedding, reverse update, and prompt handling. Existing discrete, Gaussian, and simplex branches provide concrete examples in [core/hybrid](../../core/hybrid/).

Branch-local time maps tie each physical schedule to shared global time. These maps and adapter weights belong in branch configuration; no process needs to inspect a sibling branch. The current hybrid training loop does not forward standalone adaptive-loss observations to branch schedules. A learned/adaptive branch should not be assumed trainable merely because it composes.

## Modifiers

[Self-conditioning](../modifiers/self-conditioning.md) retains **one** feedback vector for the fused model, not one per branch. Posterior-mean feedback uses the fused clean embedding. [Block diffusion](../modifiers/block-diffusion.md) applies to the complete hybrid trajectory, so all views refer to the same active block and observed prefix.

## Historical checkpoint evaluation

```bash
uv run python main.py +experiment=candi_hybrid mode=evaluate \
  evaluation.legacy=candi evaluation.checkpoint=/path/to/last.ckpt \
  evaluation.max_examples=2000 loader.eval_batch_size=64
```

The importer maps the shared embedding and parameters, validates shapes, and reconstructs EMA ordering without duplicate registrations. `evaluation.use_ema=false` selects final weights.

The imported 20,000-step Sudoku Hard CANDI checkpoint solved 1,566/2,000 puzzles (78.30%) at mean 3.503 NFE, compared with its historical 1,560/2,000 (78.00%) and 3.493 NFE. These are inference results with old weights, not corrected-mask retraining or bitwise parity.

A separate fresh 2,000-step standard-masking run solved 746/2,000 (37.3%), with 89.74% token accuracy and mean 9.1495 NFE. A fresh 2,000-step uniform-plus-simplex run solved 237/2,000 (11.85%) with 82.87% token accuracy and 181 NFE. These short checks demonstrate trainability and generation; they are not converged benchmarks. See [recorded results](../validation/reproduction-results.md).

Sources: [algorithm](../../core/algorithms/hybrid_diffusion.py), [branch implementations](../../core/hybrid/branches.py), [hybrid model](../../core/models/hybrid.py), [model configurations](../../configs/model/).
