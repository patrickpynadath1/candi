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
| Masked | absorbing process, plain cross-entropy (no ELBO weighting) |
| Uniform | uniform process, posterior cross-entropy (`ablation_path=uniform`) |
| Inference | 128 NFE per sequence; five sampler policies tuned per model on 128 validation examples x 2 seeds |
| Selection | the policy with the best validation score is scored once on 256 test examples x 2 fresh seeds (1041, 1042); ties go to the first listed policy |
| Metrics | Sudoku: share of puzzles solved; text8: unique valid words (length >= 5) per sample; TinyStories: corpus BLEU |

The five policies (the post's "Inference Pipeline"):

| Policy | Sampler | Searched |
| --- | --- | --- |
| `native` | the method's own reverse step (`HybridSampler`) | temperature, 16 trials |
| `native_greedy` | same step, argmax clean estimate | final denoise greedy/sample, 2 trials |
| `eb_greedy` | commit argmax tokens below an entropy bound (`ParallelEntropyBoundedSampler`) | temperature, entropy bound, 16 trials |
| `renoise_sample` | sample a clean sequence, re-corrupt with the forward process ([configs/policy/renoise_sample.yaml](../../configs/policy/renoise_sample.yaml)) | temperature, 16 trials |
| `renoise_greedy` | the same with the argmax sequence ([configs/policy/renoise_greedy.yaml](../../configs/policy/renoise_greedy.yaml)) | final denoise greedy/sample, 2 trials |

The sweep is [configs/sweep/sc_vs_depth.yaml](../../configs/sweep/sc_vs_depth.yaml); the
evaluation splits are in [evaluation/](evaluation/) (`preparation.evaluation_file`).

## Running it

```bash
# 1. Expand the 27 cases (no training). TinyStories needs its data and cohort, see below.
uv run --locked --extra tuning --extra scoring --extra tracking python -m core.sweeps.prepare -m \
  +sweep=sc_vs_depth task=battery/sudoku_hard,battery/text8 \
  preparation.output=$CANDI_STORAGE_ROOT/plans/sc-vs-depth
TINYSTORIES_DATA_DIR=/path/to/TinyStories/data \
uv run --locked --extra tuning --extra scoring --extra tracking python -m core.sweeps.prepare -m \
  +sweep=sc_vs_depth task=battery/tinystories_bpe data.allowed_ids_path=/path/to/allowed_ids.json \
  preparation.output=$CANDI_STORAGE_ROOT/plans/sc-vs-depth
uv run --locked --extra tuning python -m core.sweeps.prepare collect $CANDI_STORAGE_ROOT/plans/sc-vs-depth
# 2. Train and tune every case on Ray; only each case's winner sees the test split.
uv run --locked --extra tuning --extra scoring --extra tracking python -m core.sweeps.nested_tuning \
  $CANDI_STORAGE_ROOT/plans/sc-vs-depth/manifest.json \
  --output $CANDI_STORAGE_ROOT/runs/sc-vs-depth --address auto --case-concurrency 4
# 3. The post's table (writes sc_depth_results.json, compares with expected.csv).
uv run --locked --extra scoring python experiments/sc-vs-depth/analyze.py \
  $CANDI_STORAGE_ROOT/runs/sc-vs-depth --output $CANDI_STORAGE_ROOT/runs/sc-vs-depth/analysis
```

Without the TinyStories data, `+sweep=sc_vs_depth` alone prepares all 27 cases, but the
nine TinyStories cases need `TINYSTORIES_DATA_DIR` (or a Hugging Face download) and the
cohort file to run. Prepared cases always attach a W&B logger, so the workers need the
`tracking` extra (set `WANDB_MODE=offline` to keep it local).

`analyze.py` takes several run roots and merges cases of the same name, for example a
first run and a later run that tuned more samplers on the same checkpoints.
`python experiments/sc-vs-depth/analyze.py ROOT --website sc_depth_results.json` also
diffs every field against a published JSON.

## Cost

From the stored runs (training time from the callback timestamps, inference from the
recorded evaluation seconds of every tuning trial and the final test):

| Task | GPU | Train per case (8L / 8L+SC / 16L) | Inference per case | 9 cases |
| --- | --- | --- | --- | --- |
| Sudoku-hard | RTX A6000 | 1.1-1.3 h / 1.4-1.5 h / 2.0-2.2 h | 0.9-1.3 h | about 23 GPU-h |
| text8 | RTX A6000 | 1.3-1.5 h / 1.7-1.8 h / 2.6-2.7 h | 1.3-1.8 h | about 30 GPU-h |
| TinyStories BPE-10k | AMD MI210 | 3.0-3.3 h / 4.1-4.3 h / 5.3-5.6 h | 1.8-2.3 h | about 56 GPU-h |

About 110 GPU-hours in total (70 training, 40 sampler tuning), one GPU per case. SC costs
about 1.25x the 8-layer training time and 16 layers about 1.8x.

## Results

Test score of the validation-selected sampler (the post's figures; `expected.csv` has
seeds, validation scores and notes). TinyStories BLEU without the brevity penalty in
parentheses.

| Task | Method | 8L | 8L + SC | 16L |
| --- | --- | --- | --- | --- |
| Sudoku-hard (solved) | Gaussian | 0.520 renoise_greedy | **0.928** renoise_sample | 0.846 renoise_sample |
| | masked | 0.969 eb_greedy | 0.930 eb_greedy | **0.980** eb_greedy |
| | uniform | 0.826 renoise_greedy | **0.943** eb_greedy | 0.896 renoise_sample |
| text8 (valid words) | Gaussian | 12.84 renoise_sample | **14.39** native | 13.72 renoise_sample |
| | masked | **14.22** renoise_sample | 12.78 native | 13.67 renoise_sample |
| | uniform | 16.13 renoise_sample | 13.94 native | **16.55** renoise_sample |
| TinyStories (BLEU) | Gaussian | 6.81 (6.81) renoise_sample | **7.86** (7.93) native | 7.18 (7.33) renoise_sample |
| | masked | 6.67 (7.83) native | **6.91** (7.76) native | 6.77 (9.03) native |
| | uniform | 6.62 (7.45) eb_greedy | **7.37** (7.73) eb_greedy | 6.65 (7.27) renoise_sample |

SC beats doubling the depth for Gaussian corruption on every task; the discrete methods
are mixed, and on text8 SC is the worst option for both. The sampler column explains
much of that: see [sc-sampler-choice](../sc-sampler-choice/).

### Reproduction check

`analyze.py` over the stored runs (the three ruqi6000 campaigns plus the MI210 archive)
reproduces the published `sc_depth_results.json` row for row: test scores, per-seed
scores, validation scores, winners and ties, every policy in `samplers` with its seeds
and parameters, and the TinyStories BLEU parts (`score_nobp`, `seeds_nobp`, `bp`,
`len_ratio`, recomputed from the stored samples). It reports 0 differences. Two
differences in form, not results:

- `campaign` is the run directory's name. The published TinyStories rows say
  `portable-v2-mi210 (TinyStories BPE-10k)`; the directory is `portable-v2-mi210`.
- `renoise_tried` is written on every row (true when a renoise policy was tuned); the
  published file has it on the TinyStories rows only.

The MI210 run named the renoise policies `renoising` and `renoising_greedy`;
`analyze.py` reads them as `renoise_sample` and `renoise_greedy`.

## How the published numbers differ from this protocol

The sweep reproduces the Sudoku and text8 configurations exactly (checked key by key
against each stored `protocol.json`; the model config differs only in the newer
`ModularDiT` layout, whose parameter paths are unchanged). The TinyStories cells were
trained on other hardware and differ in four ways, noted per row in `expected.csv`:

1. **Masked objective.** The published masked TinyStories models used ELBO weighting
   (`KeepRateHazardWeighting`). This sweep trains masked with plain cross-entropy, as on
   Sudoku and text8. Expect different masked TinyStories numbers.
2. **Renoise implementation.** The published TinyStories renoise policies used
   `ForwardResamplingSampler` (`renoising_greedy` with `greedy_clean_estimate` and no
   initial points). This sweep uses the composed sampler with `ForwardCorruption` for
   every task, as the Sudoku and text8 runs did.
3. **Data cohort.** The published runs restricted TinyStories to a frozen cohort
   (`data.allowed_ids_path`): the stories that fit every BPE vocabulary's sequence
   length (260 at 1024, 500 at 768, 1000/5000/10000 at 512 tokens, 64-byte prompt),
   with validation stories that also occur in training removed. That leaves 1,444,319
   training and 13,092 validation stories (allowlist sha256 `f16da32a1308b16d...`).
   The indices in `evaluation/tinystories-bpe-10000.json` (the only TinyStories split
   that exists) refer to that cohort's validation order. The 108 MB allowlist is not in
   the repository; without it the indices select other stories and training sees more
   data.
4. **Hardware and source.** AMD MI210, torch 2.9.1+rocm6.3, source commit `aa4ee07` plus
   a patch; the Sudoku and text8 runs used an RTX A6000 with torch 2.5.1+cu121.

The uniform recipe is the same on both machines (`uniform_posterior_ce`).

Other caveats:

- On Sudoku and text8, the two renoise policies were tuned in a second run on the
  stored checkpoints and merged with the first run's three policies; the sweep tunes all
  five in one case. Each policy is its own Optuna study, so this does not change the
  selection.
- The Sudoku Gaussian 8L and 16L runs were resumed from step 3008 after a crash in the
  CDCD inverse CDF (the clamp fix is in `core/time_sampling/cdcd.py` now).
- Two cells are validation ties (Sudoku Gaussian 16L: both renoise policies; Sudoku
  uniform SC: `eb_greedy` and `renoise_sample`); the first listed policy was evaluated.
- One training seed per cell. Several discrete gaps are within seed noise.
