# Is self-conditioning just a deeper network?

Self-conditioning (SC) runs the network twice per training step, so part of its gain
could just be extra depth. This example compares an 8-layer model with SC against a
16-layer model without it, both against the same 8-layer baseline, for three corruption
processes on three tasks. It is the first question of the post's section
"Self Conditioning: What does it do?". The second question,
[sc-sampler-choice](../sc-sampler-choice/), reads the same runs.

"SC" means the model receives the expected embedding under its previous prediction
(`feedback=expected_embedding`, `self_conditioning.source=posterior_mean`).

## Protocol

| | |
| --- | --- |
| Grid | {Gaussian, masked, uniform} x {8L, 8L+SC, 16L} x {Sudoku-hard, text8, TinyStories BPE-10k}: 27 cases, one training seed |
| Backbone | DiT, width 512, 8 heads, 8 layers (16 for the depth arm), dropout 0.1 |
| Training | 20,000 updates, batch 256, lr 3e-4, 2,500 warmup steps, EMA 0.9999, bf16 |
| Gaussian | embedding Gaussian diffusion, CDCD loss-CDF time sampler (`ablation_path=gaussian_cdcd`) |
| Masked | absorbing process trained on the ELBO: cross-entropy with keep-rate hazard weighting (`ablation_path=masked_elbo`) |
| Uniform | uniform process, posterior cross-entropy (`ablation_path=uniform`) |
| Inference | 128 NFE per sequence; five sampler policies tuned per model on 128 validation examples x 2 seeds |
| Selection | the policy with the best validation score is scored once on 256 test examples x 2 fresh seeds (1041, 1042); ties go to the first listed policy |
| Hardware | the published runs: one AMD MI210 per case (torch 2.9.1+rocm6.3) |
| Metrics | Sudoku: share of puzzles solved; text8: unique valid words (length >= 5) per sample; TinyStories: corpus BLEU |

The five policies (the post's "Inference Pipeline"):

| Policy | Sampler | Searched |
| --- | --- | --- |
| `native` | the method's own reverse step (`HybridSampler`) | temperature, 16 trials |
| `native_greedy` | same step, argmax clean estimate | final denoise greedy/sample, 2 trials |
| `eb_greedy` | commit argmax tokens below an entropy bound (`ParallelEntropyBoundedSampler`) | temperature, entropy bound, 16 trials |
| `renoise_sample` | sample a clean sequence, re-corrupt it with the forward process to the next level (`ForwardResamplingSampler`) | temperature, 16 trials |
| `renoise_greedy` | the same with the argmax clean estimate | final denoise greedy/sample, 2 trials |

The policies are [configs/inference_policies/hybrid_renoising.yaml](../../configs/inference_policies/hybrid_renoising.yaml),
where the two renoise policies are named `renoising` and `renoising_greedy`; `analyze.py`
reports them as `renoise_sample` and `renoise_greedy`, the post's names. The sweep is
[configs/sweep/sc_vs_depth.yaml](../../configs/sweep/sc_vs_depth.yaml); the evaluation splits are in [evaluation/](evaluation/) (`preparation.evaluation_file`).

## Running it

```bash
# 0. Data the published runs used that is too large to commit (see "How a rerun can differ").
python scripts/prepare_data.py text8 --parquet PATH/afmck-text8/data \
  --raw-dir $SDDC_STORAGE_ROOT/data_cache/text8/raw
python scripts/prepare_data.py tinystories --parquet PATH/roneneldan-TinyStories/data \
  --output $SDDC_STORAGE_ROOT/data_cache/tinystories_bpe/allowed_ids.json
# 1. Expand the 27 cases (no training).
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=sc_vs_depth preparation.output=$SDDC_STORAGE_ROOT/plans/sc-vs-depth
uv run --locked --extra tuning python -m core.sweeps.prepare collect $SDDC_STORAGE_ROOT/plans/sc-vs-depth
# 2. Train and tune every case on Ray; only each case's winner sees the test split.
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/sc-vs-depth/manifest.json \
  --output $SDDC_STORAGE_ROOT/runs/sc-vs-depth --address auto --case-concurrency 4
# 3. The post's table (writes sc_depth_results.json, compares with expected.csv).
uv run --locked --extra scoring python experiments/sc-vs-depth/analyze.py \
  $SDDC_STORAGE_ROOT/runs/sc-vs-depth --output $SDDC_STORAGE_ROOT/runs/sc-vs-depth/analysis
```

TinyStories reads its stories from `TINYSTORIES_DATA_DIR` (or a Hugging Face download)
and the cohort from `${data.cache_dir}/allowed_ids.json`. W&B logging is off unless
`WANDB_ENTITY` is set (with `--extra tracking`).

`analyze.py` takes several run roots and merges cases of the same name, for example a
first run and a later run that tuned more samplers on the same checkpoints.
`python experiments/sc-vs-depth/analyze.py ROOT --website sc_depth_results.json` also
diffs every field against a published JSON.
## Cost

From the stored runs (`wall_hours_approx_*` in the archive's summary, from file
timestamps), one AMD MI210 per case:

| Task | Train per case (8L / 8L+SC / 16L) | Inference per case | 9 cases |
| --- | --- | --- | --- |
| Sudoku-hard | 1.1-1.3 h / 1.4-1.5 h / 2.0-2.2 h | 0.9-1.3 h | about 23 GPU-h |
| text8 | 1.3-1.5 h / 1.7-1.8 h / 2.6-2.7 h | 1.3-1.8 h | about 30 GPU-h |
| TinyStories BPE-10k | 3.0-3.3 h / 4.1-4.3 h / 5.3-5.6 h | 1.8-2.3 h | about 56 GPU-h |

About 110 GPU-hours in total (70 training, 40 sampler tuning). SC costs about 1.25x the
8-layer training time and 16 layers about 1.8x.
## Results

Test score of the validation-selected sampler (the post's figures; `expected.csv` has
seeds, validation scores and notes). TinyStories BLEU without the brevity penalty in
parentheses.

| Task | Method | 8L | 8L + SC | 16L |
| --- | --- | --- | --- | --- |
| Sudoku-hard (solved) | Gaussian | 0.533 renoise_sample | **0.906** renoise_greedy | 0.807 renoise_sample |
| | masked | 0.969 eb_greedy | 0.926 eb_greedy | **0.973** eb_greedy |
| | uniform | 0.836 renoise_sample | **0.928** renoise_greedy | 0.906 renoise_sample |
| text8 (valid words) | Gaussian | 12.96 renoise_sample | **14.07** native | 13.65 renoise_sample |
| | masked | **13.90** renoise_sample | 13.49 native | 13.85 renoise_sample |
| | uniform | 16.30 renoise_sample | 14.33 native | **16.54** renoise_sample |
| TinyStories (BLEU) | Gaussian | 6.81 (6.81) renoise_sample | **7.86** (7.93) native | 7.18 (7.33) renoise_sample |
| | masked | 6.67 (7.83) native | **6.91** (7.76) native | 6.77 (9.03) native |
| | uniform | 6.62 (7.45) eb_greedy | **7.37** (7.73) eb_greedy | 6.65 (7.27) renoise_sample |

SC beats doubling the depth for Gaussian corruption on every task; the discrete methods
are mixed, and on text8 SC is the worst option for both. The sampler column explains
much of that: see [sc-sampler-choice](../sc-sampler-choice/).

### Reproduction check

`analyze.py` over the stored runs (the MI210 archive `portable-v2-mi210-partial-20261106-0013`)
reproduces the published `sc_depth_results.json` row for row: test scores, per-seed
scores, validation scores, winners and ties, every policy in `samplers` with its seeds
and parameters, and the TinyStories BLEU parts (`score_nobp`, `seeds_nobp`, `bp`,
`len_ratio`, recomputed from the stored samples). It reports 0 differences.

Every resolved config this sweep prepares was compared key by key with the stored
`protocol.json` of its published case, for all 27 cases: training, objective weighting,
SC, time sampler, samplers, search spaces, budgets, evaluation indices and seeds match.
The remaining differences are machine paths, the newer `ModularDiT` model layout (same
architecture), and two settings applied at evaluation time through the evaluation file
instead of the task config (`task.num_eval_samples`, text8's `task.min_word_lengths`).
## How a rerun can differ

- **Code and hardware.** The published runs used source commit `aa4ee07` plus a patch,
  torch 2.9.1+rocm6.3 on MI210s. The model configs have since moved to `ModularDiT`
  slots, and the training loader now draws a per-epoch shuffled order of whole optimizer
  steps with a separately seeded validation RNG. A rerun follows a different training
  trajectory; expect close numbers, not equal ones.
- **text8 source.** The published runs used the Hugging Face `afmck/text8` parquet copy,
  written out as the task's 90M/5M/5M character split files; `prepare_data.py text8`
  rebuilds them so the evaluation indices pick the same validation windows.
- **TinyStories cohort.** The evaluation indices refer to a story cohort shared with the
  [tokenizer ladder](../tokenizer-ladder/) (1,444,319 train and 13,092 validation
  stories); `prepare_data.py tinystories` rebuilds it, and the sweep reads it from
  `${data.cache_dir}/allowed_ids.json`. Without it the indices select other stories.
- **One training seed per cell.** Several discrete gaps are within seed noise.
- **Ties.** Sudoku uniform 16L is a validation tie between the two renoise policies; the
  first listed policy was evaluated.
- **Sampler collapses.** Some renoise and entropy-bounded policies score near zero on
  some checkpoints (for example, renoise after SC on text8 masked: under one valid word
  per sample). Validation selection is unaffected, since another policy wins there.
