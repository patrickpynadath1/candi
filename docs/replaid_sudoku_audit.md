# Small RePlaid Sudoku baseline

**Superseding correction:** the initial pilots and first full-size runs below
had a CUDA autocast/self-conditioning bug that prevented transformer updates.
Their performance numbers are invalid as RePlaid baselines. See the final
section for the diagnosis and replacement run.

Branch: `codex/small-scale-replaid`. Base commit:
`0a1ff20ed01fc2b46ef67bbf29471784f6a49c8a`.

The repository already contained RePlaid. This branch audits and corrects that
implementation, then measures a small conditional Sudoku model trained from
scratch. This is not a reproduction of the paper's language benchmarks.

## Sources and audit

- [RePlaid equation 4 and Appendix B](https://arxiv.org/html/2605.18530v1#S2):
  reconstruction CE, terminal Gaussian KL, and SNR-derivative-weighted embedding
  MSE; categorical predictions are projected through learned unit embeddings.
- [Appendix I](https://arxiv.org/html/2605.18530v1#A9): shape parameters minimize
  the diffusion estimator's second moment; endpoints minimize the ordinary loss.
- [Published Plaid model](https://github.com/igul222/plaid/blob/main/lib/models.py):
  positive-weight tanh schedule normalized at zero and one, and second-moment
  scaling of noisy inputs and self-conditioning embeddings.

Changes and checks:

1. The default schedule is now a 1,024-unit positive-weight tanh MLP. The previous
   positive softplus basis was restricted to convex curves. An explicit `legacy`
   option retains it for the existing RePlaidKeep configurations. Endpoint
   parameters are independent, as in published Plaid. Their ordering is checked
   in the saved-run audit; optimization does not constrain their ordering.
2. Both normalized shape outputs (value and time derivative) receive the `2*loss`
   gradient hook. Tests compare resulting shape gradients against direct
   differentiation of the second moment, and endpoint gradients against the
   mean loss. Reconstruction rows do not update the shape. Self-conditioned
   rows detach embedding and schedule paths in the denoising loss.
3. Analytic schedule derivatives match autograd in float64. Noise obeys the VP
   identity; the loss sums embedding dimensions before averaging tokens.
4. Sudoku's solution mask is used for schedule variance and batch-allocation
   statistics. Running first and second moments replace an EMA of within-batch
   standard deviations, which incorrectly gave zero variance for singleton
   reconstruction sub-batches.
5. Noisy inputs are divided by `sqrt(alpha²/d_e + sigma²)`; self-conditioning
   embeddings are multiplied by `sqrt(d_e)`. Explicit autocast boundaries keep
   input projections, Gaussian logits, and embedding expectations in FP32 while
   the transformer/head use BF16 and schedule/loss weighting use FP64.
6. A singleton data batch now includes both reconstruction and diffusion draws.
   An analytic uniform-predictor test checks the complete three-term objective.
7. DDPM includes posterior noise in the final transition. Disabling
   self-conditioning disables its validation bootstrap and sampling feedback.
   Sampling now bootstraps at each current state as described in Appendix E.2;
   both denoiser calls count toward NFE. `sampling_self_condition=previous`
   retains the previous sampler for explicit comparisons. The DDPM appendix's
   one-call accounting does not include this extra bootstrap, so we report
   actual calls rather than treating transitions as NFE.
8. Optimizer parameter coverage, checkpoint state roundtrip, prompt preservation,
   and real CUDA precision boundaries have regression tests.

## Conditional task and deliberate differences

`algo.clean_context=true` keeps the puzzle prefix clean in training and sampling;
only solution tokens enter the loss. This is a conditional adaptation, whereas
the paper's main experiments are unconditional. The default algorithm config
keeps this option off. The legacy noisy-prompt sampler remains available.

The experiment uses a four-layer, width-128 DiT (~1.3M parameters), embedding
dimension 16, batch size 128, 2,000 optimizer steps, seed 1, 100 warmup steps,
EMA 0.999, and no dropout. Initial negative-log-SNR endpoints are -13.3 and 5.0.
Backbone LR is 0.0003; embedding, schedule-shape, and endpoint LR is 0.01.
Only endpoints have weight decay (0.1). These are small-experiment choices,
not a claim of identical paper hyperparameters or training budget.

Dataset: cached `sudoku_hard_train200000_valid10000_seed42`; the generator targets
30 clues and checks uniqueness. Sequences have 180 tokens: 91 puzzle-prefix
tokens and 89 solution tokens, including eight row separators. Training sees
256,000 examples (22,784,000 scored solution tokens).

The pilot callback evaluated a fixed, randomly selected 128-example validation subset
every 500 steps with 180 DDPM transitions and previous-prediction self-conditioning
(181 calls). The final standalone evaluation uses the corrected fresh-bootstrap
sampler (362 calls). `cell_accuracy`
includes row separators. The standalone evaluation additionally reports digit
accuracy, clue consistency, and valid solutions. Validation NELBO uses eight
batches (1,024 examples) and is a Monte Carlo estimate.

## Reproduce

From the repository root, select an available GPU (GPU 3 was used here):

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 .venv/bin/python main.py \
  +experiment=sudoku_replaid_small \
  hydra.run.dir=/home/patrick/duo/outputs/replaid_small_sudoku_audited_seed1 \
  +trainer.enable_progress_bar=false
```

Use a fresh run directory for a new experiment; automatic resume is disabled.
The resolved Hydra config, CSV metrics, Sudoku JSON records, and checkpoints
are saved under that directory. A preliminary run with the old schedule is
retained under `outputs/replaid_small_sudoku_seed1` and is not the audited result.

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 .venv/bin/python main.py \
  +experiment=sudoku_replaid_small mode=sudoku_eval seed=123 \
  eval.checkpoint_path=/home/patrick/duo/outputs/replaid_small_sudoku_audited_seed1/checkpoints/last.ckpt \
  sudoku.max_examples=512 \
  sudoku.output_dir=/home/patrick/duo/outputs/replaid_small_sudoku_audited_seed1/eval512_bootstrap \
  hydra.run.dir=/home/patrick/duo/outputs/replaid_small_sudoku_audited_seed1/eval512_bootstrap
```

The new schedule and additional running-moment buffers change checkpoint state.
Older RePlaid checkpoints are not interchangeable with this experiment config.

## Pilot results

The audited 2,000-step pilot completed. Final EMA validation NELBO was
2.474566 nats per solution token. Fresh-bootstrap evaluation on 512 held-out
puzzles solved **0/512**, with digit accuracy **8.50%** and valid-grid rate **0%**.
This pilot does not demonstrate Sudoku-solving ability and is not a matched
comparison to the much larger existing models.

The saved raw schedule is finite and strictly increasing on a 1,001-point grid:
learned gamma endpoints **-1.720984, 3.205829**, minimum derivative **1.844982**,
maximum analytic-versus-autograd derivative error **7.11e-15**. Full diagnostics
are in `outputs/replaid_small_sudoku_audited_seed1/schedule_audit.json`.

## Matched full-size run

`sudoku_replaid_matched` matches the actual saved hard-Sudoku MDLM configuration
from the September 22 campaign, rather than the older 20k-step defaults:

- Eight layers, hidden width 512, eight heads, conditioning width 128, dropout 0.1.
- Batch 256 (four microbatches of 64), 50,000 optimizer steps, seed 1.
- Same cached 200,000 training / 10,000 validation puzzles from that campaign.
- Backbone AdamW LR 0.0003, betas (0.9, 0.999), warmup 2,500, EMA 0.9999.
- Monitoring every 1,000 steps on 128 puzzles with 45 transitions; checkpoint
  every 5,000 steps. After training, the launcher evaluates all 10,000 validation
  puzzles with 180 transitions. Fresh bootstrap gives 92 and 362 calls respectively.
- RePlaid retains its 16-dimensional embeddings, learned schedule, and separate
  optimizer groups. Clean-puzzle conditioning is enabled. Native W&B logging
  is enabled in project `rz-text-dlp/replaid-sudoku`.

Run directory: `outputs/replaid_sudoku_matched_50k_20260929_seed1`.
It contains `launch.json`, `source_hashes.json`, a frozen `source/`, the saved
reference config, `status.txt`, and `train.log`. Training runs on GPU 3 in tmux
session `replaid-sudoku-50k-20260929`; `scripts/train_replaid_sudoku.sh` queues the
full evaluation after successful training. This run was launched, not completed,
when this audit was written.

Remaining reproduction differences are the conditional task, benchmark-matched
model/training settings, FP64 schedule parameter storage (the paper stores FP32
and computes FP64), and running-moment initialization. Exact author-code parity
is not claimed: the schedule was checked against public Plaid and the RePlaid
equations, not an official RePlaid implementation.

### W&B restart

At the user's request, the initial local-logging-only full-size run was stopped
and preserved. Training restarted **from scratch**, without loading its checkpoint,
in `outputs/replaid-hard-50k-20260929-wandb`, on GPU 3. The new persistent tmux
session is `replaid-sudoku-wandb-20260929`. The matched experiment now enables
the native Lightning W&B logger by default, following prior Sudoku runs.

W&B project: `rz-text-dlp/replaid-sudoku`; run ID:
`replaid-hard-50k-20260929-wandb`. No CSV backfill process is used.

### Frozen-transformer diagnosis and correction

The 5,000-step checkpoint from the W&B run had an exactly zero output-head
weight matrix and exactly zero first-block AdaLN gates. Embeddings and schedule
parameters had changed, masking the absence of transformer learning.

The CUDA reproduction isolated the interaction between Lightning's outer BF16
autocast scope and the no-grad self-conditioning bootstrap. The bootstrap's
cached BF16 transformer weights were reused without autograd in the trainable
pass. With self-conditioning off, head gradients existed; with it on, head
gradients were `None`. Disabling caching only in an outer probe was insufficient
because the model's inner context enabled it again.

The transformer autocast context now explicitly sets `cache_enabled=False`.
A real-CUDA regression test runs bootstrap and training inside an outer autocast
scope and verifies both head and attention weights change after optimizer steps.
All 26 focused CUDA tests passed. RePlaid now fails immediately if the output head
has no gradient and logs head gradient/weight norms, all three loss components,
schedule endpoints, and reconstruction allocation to W&B.

The defective run was stopped and preserved. Replacement training starts from
scratch using the same matched 28.6M/50k configuration, GPU 3, with a fresh frozen
source snapshot at `outputs/replaid-hard-50k-20260929-autocast-fix`.
Its W&B ID is `replaid-hard-50k-20260929-autocast-fix` in `rz-text-dlp/replaid-sudoku`;
tmux session: `replaid-sudoku-fixed-20260929`.

Live W&B verification through optimizer step 380 confirms nonzero head gradients
and head weight norm 0.156 (previously exactly zero after 5,000 steps). Mean
training loss over the first ten logged observations was 2.502; over the latest
ten it was 2.277. This verifies early learning after the fix, not final Sudoku
performance. The objective remains a noisy Monte Carlo estimate.
