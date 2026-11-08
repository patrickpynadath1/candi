# Does self-conditioning change how the model should be sampled?

In [sc-vs-depth](../sc-vs-depth/), self-conditioning (SC) looks like it hurts the
discrete models on text8. This example asks whether that is a property of the model
or of the sampler: which of the five sampler policies wins for each model, and what the
SC / 8-layer / 16-layer comparison looks like when every model uses the same sampler.

## Analysis only

This example has no sweep and trains nothing. Every sc-vs-depth case already tunes all
five policies (`native`, `native_greedy`, `eb_greedy`, `renoise_sample`,
`renoise_greedy`) on the validation split and records each policy's best score, so the
question is answered by reading those validation scores. A separate sweep would retrain
the same 27 checkpoints under the same protocol. Everything here uses validation scores
only; the test split is touched once per case, by the winner, inside sc-vs-depth.

```bash
# After running sc-vs-depth (see its README):
uv run --locked python experiments/sc-sampler-choice/analyze.py \
  $CANDI_STORAGE_ROOT/runs/sc-vs-depth --output $CANDI_STORAGE_ROOT/runs/sc-vs-depth/analysis
```

`analyze.py` takes the same run roots as `experiments/sc-vs-depth/analyze.py` (and
reuses its loader, including the merge of runs that tuned different policies on one
checkpoint). It prints:

1. the winning policy for each of the 27 models (the post's SamplerChoice winner map);
2. the validation score of every policy against `expected.csv` (the "All samplers"
   grid, 135 values, with the winner of each model);
3. the 8L / 8L+SC / 16L validation scores when every model is restricted to `native`.

It writes `sampler_choice.json` with the same rows. The cost is that of sc-vs-depth.

## Results

Winning policy by validation score (from the stored runs; `*` marks a tie, the first
listed policy was kept):

| Task | Method | 8L | 8L + SC | 16L |
| --- | --- | --- | --- | --- |
| Sudoku-hard | Gaussian | renoise_sample | renoise_greedy | renoise_sample |
| | masked | eb_greedy | eb_greedy | eb_greedy |
| | uniform | renoise_sample | renoise_greedy | renoise_sample* |
| text8 | Gaussian | renoise_sample | native | renoise_sample |
| | masked | renoise_sample | native | renoise_sample |
| | uniform | renoise_sample | native | renoise_sample |
| TinyStories BPE-10k | Gaussian | renoise_sample | native | renoise_sample |
| | masked | native | native | native |
| | uniform | eb_greedy | eb_greedy | renoise_sample |

On text8, renoising wins for every model without SC (8 and 16 layers) and for none of
the SC models. With SC, renoising stops working: the masked SC model gets 0.64 valid words
per sample with `renoise_sample` and 0.03 with `renoise_greedy`, against 13.64 with
`native`.

Restricted to `native`, the text8 picture flips (validation, valid words per sample):

| Method | 8L | 8L + SC | 16L |
| --- | --- | --- | --- |
| Gaussian | 11.17 | **14.16** | 12.19 |
| masked | 11.48 | **13.64** | 12.36 |
| uniform | 13.63 | 14.33 | **14.73** |

These are the numbers the post quotes (13.6 against 11.5 and 12.4 for masked; 14.3
against 13.6 and 14.7 for uniform). `analyze.py` on the stored runs reproduces all 135
policy scores and all 27 winners in `expected.csv`.

## Caveats

- The comparison is on validation scores (128 examples x 2 seeds per policy), one
  training seed per model, and the flip is on one task.
- The rows inherit sc-vs-depth's caveats (code and hardware, data, one training seed);
  see [its README](../sc-vs-depth/README.md#how-a-rerun-can-differ).
- Why SC models cannot use renoising is untested. One guess: renoising discards the noisy
  state the SC feedback was computed from.
