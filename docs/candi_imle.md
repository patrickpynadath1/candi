# Conditional IMLE-style CANDI

`algo=candi_imle` uses the standard CANDI backbone as a single-pass conditional
generator. Training starts target positions from independent Gaussian noise,
not corrupted versions of their true tokens. Visible prompt positions remain
clean one-hot anchors. The noise standard deviation is `algo.imle_noise_std=1`;
the model receives a fixed full-corruption condition.

For each example, draw `algo.imle_candidates=8` independent noise tensors,
run the same model on each, and compute per-token cross-entropy against the
target. Choose the candidate with the lowest mean CE over valid target tokens.
Only that candidate's complete sequence loss receives gradient. Candidates are
selected separately per example, never independently per position. K=1 is the
ordinary noise-conditioned supervised baseline.

The implementation retains all K forward graphs. This ensures the selected
loss is differentiated using exactly the dropout realization that was scored;
it avoids a stochastic second forward pass after selection. Forward/backward
work and activation memory scale with K. The hard-Sudoku preset uses 16-way
gradient accumulation to keep global batch 256 with smaller microbatches.

Validation uses one noise draw per example, not target-based best-of-K
selection. The shared trainer's `nll`/`ppl` metric names remain historical names
for a CE surrogate, not estimates of the generator's marginal likelihood.
During training, the metric measures best-of-K CE; it should not be compared
directly with ordinary single-sample CE without accounting for selection.

`sampling.predictor=candi_imle` makes one forward pass from fresh noise and
preserves prompt tokens. Greedy decoding is enabled in the Sudoku preset;
categorical decoding is available through `sampling.greedy_clean_estimate=false`.
There is no ground truth, target-based selection, recurrence, EB acceptance,
or iterative denoising at inference. `sampling.steps` does not control this
single-pass sampler; NFE is always one. Requesting multiple generated examples
provides independent draws without choosing among them using targets.

## Run

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python main.py \
  +experiment=sudoku_candi_imle trainer.devices=1 loader.num_workers=0 \
  data.cache_dir=/home/patrick/.cache/duo/data_cache \
  checkpointing.resume_from_ckpt=false \
  +trainer.check_val_every_n_epoch=null
```

The useful first comparison is K=1 versus K=8, reporting single-draw solve rate
and training compute. A best-of-K training loss can improve because a lucky
candidate wins even if average sample quality does not. On unique-solution
Sudoku, there is no need to represent multiple correct answers for a given
puzzle, so additional latent diversity may or may not help optimization.

## Literature

- [Li and Malik, Implicit Maximum Likelihood Estimation (2018)](https://arxiv.org/abs/1809.09087)
  matches each data example to a nearby generated sample, with connections to
  maximum likelihood under specific conditions.
- [Li, Zhang, and Malik, Diverse Image Synthesis from Semantic Layouts via
  Conditional IMLE (2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Diverse_Image_Synthesis_From_Semantic_Layouts_via_Conditional_IMLE_ICCV_2019_paper.html)
  extends nearest-sample matching to conditional generation.

This variant matches distributions to token targets using cross-entropy,
and draws a fresh finite candidate set for each example on every update.
It is an IMLE-inspired best-of-K objective, not a reproduction of those papers
or a claim that their maximum-likelihood guarantees apply to this loss.
