# Does the tokenizer change the picture?

Small diffusion-LM experiments usually model characters or bytes; real language models
use subword vocabularies. With bytes a noisy position has 260 candidates, with a 10k BPE
vocabulary it has 10,000, mostly whole words. This example repeats the method comparison
on TinyStories at five BPE vocabularies to see whether self-conditioning's (SC) effect and
the differences between methods survive the change. An autoregressive (AR) model of the
same size is the reference. It is the post's "Does the tokenizer change the picture?" figure
(`VocabLadder`) and the length panel of "A note on BLEU" (`BleuLength`).

## Protocol

35 cases, one training seed each:

| Axis | Values |
| --- | --- |
| Vocabulary (`bpe.vocabulary_size`) | 260 (bytes plus 4 specials), 500, 1000, 5000, 10000 |
| Recipe (`ladder_recipe`) | `gaussian`, `masked`, `uniform` (each without SC), `gaussian_sc`, `masked_sc`, `uniform_sc`, `ar` |

- **Data.** TinyStories at revision `f54c09fd`. The tokenizers are in
  `artifacts/tokenizers/tinystories-bpe/`. Each story is split into a 64-byte prompt
  and the continuation the model has to write. Every vocabulary uses the same stories.
  The shared TinyStories cohort (`data.allowed_ids_path`, the same file as in
  [block-encoder-decoder](../block-encoder-decoder/)) keeps a story only if it fits the
  sequence capacity of all five tokenizers. It also drops validation stories that appear in training
  (1,444,319 training and 13,092 validation stories). The sequence capacity is 1024 tokens
  at 260, 768 at 500 and 512 above (`bpe.capacity` in the sweep).
- **Models.** An 8-block DiT (hidden size 512, 8 heads, conditioning size 128, AdaLN,
  dropout 0.1) for every diffusion recipe, and an 8-block causal transformer of the same
  width for AR.
- **Recipes.** Gaussian diffusion on learned unit-norm 128-d token embeddings, with a
  CDCD-fitted time sampler and plain cross-entropy. Masked (absorbing) diffusion trained
  on the ELBO (`objective=masked_elbo`, the keep-rate hazard weighting). Uniform-noise
  diffusion with a posterior cross-entropy. SC is posterior-mean self-conditioning (the
  expected embedding of the previous prediction), used in 90% of training steps.
- **Training.** 20,000 steps, batch 256 (4 accumulation steps), learning rate 3e-4
  with 2,500 warmup steps, then constant, EMA 0.9999, bf16 mixed precision. No training
  search.
- **Inference.** Every policy is tuned with Optuna on 128 validation stories (seeds 41
  and 42). Only the validation winner generates the 256 test stories (seeds 1041 and
  1042). The score is corpus BLEU (sacrebleu) against the real continuation. Diffusion
  gets 128 model calls per story. AR makes one call per token, so its budget is the
  sequence capacity. The indices and seeds are in [`evaluation/`](evaluation/).

| Policy (site name) | Sampler | Searched | Trials |
| --- | --- | --- | ---: |
| `native` | the recipe's ancestral hybrid sampler | temperature in [0.5, 1.5] | 16 |
| `native_greedy` | same, greedy clean estimate | final denoise: greedy or sample | 2 |
| `eb_greedy` | parallel entropy-bounded, greedy, frozen | temperature, entropy bound in [0.02, 0.5] (log) | 16 |
| `renoising` (`renoise_sample`) | forward re-noising, sampled clean estimate | temperature | 16 |
| `renoising_greedy` (`renoise_greedy`) | forward re-noising, greedy clean estimate | final denoise | 2 |
| AR `native` | cached sequential decoding, sampling | temperature | 16 |
| AR `native_greedy` | cached sequential decoding, greedy | none | 1 |

The configs: [`configs/sweep/tokenizer_ladder.yaml`](../../configs/sweep/tokenizer_ladder.yaml),
`configs/ladder_recipe/*.yaml`, `configs/inference_policies/{hybrid_renoising,autoregressive}.yaml`
and `configs/objective/masked_elbo.yaml`.

## Running it

```bash
export CANDI_STORAGE_ROOT=~/.cache/candi   # or your storage volume
EXTRAS=(--extra tuning --extra scoring --extra tracking)

# 0. TinyStories parquet shards, then the shared story cohort. The script exits
#    non-zero unless its sha256 is f16da32a..., the cohort the committed indices use.
huggingface-cli download roneneldan/TinyStories --repo-type dataset \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 --include 'data/*' \
  --local-dir $CANDI_STORAGE_ROOT/tinystories
export TINYSTORIES_DATA_DIR=$CANDI_STORAGE_ROOT/tinystories/data
uv run --locked "${EXTRAS[@]}" python experiments/block-encoder-decoder/prepare_data.py tinystories \
  --parquet $TINYSTORIES_DATA_DIR \
  --output $CANDI_STORAGE_ROOT/data_cache/tinystories_bpe/allowed_ids.json

# 1. Expand the sweep into 35 cases (no training).
uv run --locked "${EXTRAS[@]}" python -m core.sweeps.prepare -m \
  +sweep=tokenizer_ladder preparation.output=$CANDI_STORAGE_ROOT/plans/tokenizer-ladder
uv run --locked "${EXTRAS[@]}" python -m core.sweeps.prepare collect \
  $CANDI_STORAGE_ROOT/plans/tokenizer-ladder
# 2. Train each case, tune its samplers on validation, test the winner.
uv run --locked "${EXTRAS[@]}" python -m core.sweeps.nested_tuning \
  $CANDI_STORAGE_ROOT/plans/tokenizer-ladder/manifest.json \
  --output $CANDI_STORAGE_ROOT/runs/tokenizer-ladder --address auto --case-concurrency 4
# 3. The post's table (vocab_ladder.json) and a comparison with expected.csv.
uv run --locked "${EXTRAS[@]}" python experiments/tokenizer-ladder/analyze.py \
  $CANDI_STORAGE_ROOT/runs/tokenizer-ladder --output results/tokenizer-ladder
```

Prepared cases always configure a W&B logger, so install the `tracking` extra. Set
`WANDB_ENTITY` to log online, or `WANDB_MODE=offline`/`disabled` to keep it local. Run
one vocabulary with `bpe.vocabulary_size=1000`, one recipe with `ladder_recipe=masked_sc`.

The AR cases tune `algorithm.predictor.commitment` (sample or greedy). Until
`core.sweeps.nested_tuning.validate_inference` accepts that key, preparing them fails
with "Not a tunable inference parameter"; the 30 diffusion cases prepare either way.

## Cost

Wall-clock hours on one AMD MI210 per case, from the stored runs' file timestamps:
142 GPU-hours of training and 195 in total, with tuning and the test run. By
vocabulary (training / total): 260: 42 / 54, 500: 31 / 42, 1000: 21 / 30, 5000: 23 / 33,
10000: 25 / 37. One case takes 2.6 to 7 hours to train and 3 to 9 hours in total. The
cases with the longest sequences (260 tokens, 1024 positions) and SC cost the most.
Tuning runs one trial at a time per case.

## Results

The published numbers, reproduced by `analyze.py` from the stored runs. Test BLEU is
the chosen sampler's mean over the two test seeds. "No BP" divides out the brevity
penalty. BP and the length ratio (generated / reference length) are seed means.

| Vocab | Method | SC | Chosen sampler | Test BLEU | BLEU, no BP | BP | Length ratio |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 260 | AR | n/a | native_greedy | 8.885 | 9.328 | 0.953 | 0.954 |
| 500 | AR | n/a | native_greedy | 8.917 | 9.155 | 0.974 | 0.974 |
| 1000 | AR | n/a | native_greedy | 9.067 | 9.067 | 1.000 | 1.025 |
| 5000 | AR | n/a | native_greedy | 9.682 | 9.820 | 0.986 | 0.986 |
| 10000 | AR | n/a | native | 8.946 | 8.973 | 0.997 | 1.004 |
| 260 | Gaussian | no | renoise_sample | 6.009 | 7.168 | 0.838 | 0.850 |
| 500 | Gaussian | no | renoise_sample | 6.527 | 6.529 | 1.000 | 1.001 |
| 1000 | Gaussian | no | renoise_sample | 6.910 | 6.940 | 0.996 | 0.997 |
| 5000 | Gaussian | no | renoise_sample | 7.140 | 7.172 | 0.995 | 0.999 |
| 10000 | Gaussian | no | renoise_sample | 6.812 | 6.812 | 1.000 | 1.027 |
| 260 | Gaussian | yes | native | 7.550 | 8.158 | 0.925 | 0.928 |
| 500 | Gaussian | yes | native | 7.482 | 7.482 | 1.000 | 1.006 |
| 1000 | Gaussian | yes | renoise_sample | 7.582 | 7.590 | 0.999 | 1.004 |
| 5000 | Gaussian | yes | native | 7.657 | 8.133 | 0.941 | 0.943 |
| 10000 | Gaussian | yes | native | 7.864 | 7.934 | 0.991 | 0.994 |
| 260 | Masked | no | native_greedy | 6.209 | 8.273 | 0.750 | 0.777 |
| 500 | Masked | no | native | 5.918 | 7.149 | 0.828 | 0.841 |
| 1000 | Masked | no | native | 6.436 | 8.112 | 0.793 | 0.812 |
| 5000 | Masked | no | native | 6.456 | 7.041 | 0.917 | 0.920 |
| 10000 | Masked | no | native | 6.671 | 7.833 | 0.852 | 0.862 |
| 260 | Masked | yes | native_greedy | 6.775 | 9.228 | 0.734 | 0.764 |
| 500 | Masked | yes | native | 6.809 | 7.612 | 0.894 | 0.900 |
| 1000 | Masked | yes | native | 7.114 | 9.227 | 0.771 | 0.794 |
| 5000 | Masked | yes | native | 7.250 | 8.273 | 0.876 | 0.883 |
| 10000 | Masked | yes | native | 6.908 | 7.762 | 0.890 | 0.896 |
| 260 | Uniform | no | eb_greedy | 6.271 | 6.271 | 1.000 | 1.016 |
| 500 | Uniform | no | renoise_sample | 7.011 | 7.133 | 0.983 | 0.984 |
| 1000 | Uniform | no | renoise_sample | 7.279 | 7.294 | 0.998 | 0.998 |
| 5000 | Uniform | no | native | 6.399 | 7.671 | 0.834 | 0.847 |
| 10000 | Uniform | no | eb_greedy | 6.624 | 7.453 | 0.889 | 0.895 |
| 260 | Uniform | yes | native | 7.509 | 7.529 | 0.997 | 1.001 |
| 500 | Uniform | yes | renoise_sample | 7.408 | 8.310 | 0.892 | 0.897 |
| 1000 | Uniform | yes | eb_greedy | 7.235 | 7.345 | 0.985 | 0.985 |
| 5000 | Uniform | yes | eb_greedy | 7.574 | 8.016 | 0.945 | 0.947 |
| 10000 | Uniform | yes | eb_greedy | 7.367 | 7.732 | 0.953 | 0.954 |

`vocab_ladder.json` also holds every policy's validation score per seed and its tuned
parameters. On the stored runs, `analyze.py` matches the website's `vocab_ladder.json`
in every field of all 35 rows: chosen sampler, validation scores and parameters, test
scores, and BLEU with and without the brevity penalty. Only the free-text `source` line
differs, because the hardware and run names are not in the run tree. Masked models end
their stories early (length ratio 0.76 to 0.92) and lose the most to the penalty.

## The judge-model study (not part of this example)

The post's "A second opinion from a judge model" section asked an LLM judge,
[Jev](https://docs.typesafe.ai/concepts/system-one) from TypeSafe, to read the ladder's
test stories: the chosen sampler's 512 continuations per model (256 prompts x 2 test
seeds), about 160,000 typed questions. The questions:

- **Best of three.** Gaussian, masked or uniform (all with SC, or all without).
- **SC or not.** The same method with and without SC.
- **Real or model.** Which ending the original author wrote.
- **Ratings.** Coherence, fluency and topicality, divided by the real endings' scores.
- **Garbled words.** Whether each distinct word is misspelled, made up, a fragment or
  two fused words.

Every pairwise question was asked in both orders and averaged. The results are in the
website's `jev_ladder.json`. Jev is a paid external API and the analysis script is
private, so neither is in this repository. The test records that `analyze.py` reads
(`final/seed-*/results.json`, fields `prompt`, `continuation`, `reference`) are all such
a study needs.

## Caveats and differences from the published runs

- **Hardware and code.** The published runs trained on AMD MI210 GPUs with torch
  2.9.1+rocm6.3, from an earlier source revision. Since then the model configs moved to
  ModularDiT slots (`model.time.adaLN`, `model.self_conditioning`, ...). The architecture
  should be the same, but this was checked only by comparing configs. The training
  loader now draws a per-epoch shuffled order of whole optimizer steps, and the
  validation RNG is seeded separately. A rerun therefore follows a different training
  trajectory and will not match the post exactly.
- **What was checked.** Apart from that model translation and machine paths, every
  resolved config prepared from this sweep matches its stored run, key for key, for all
  35 cases: training, capacities, objective weighting, SC, time sampler, samplers,
  search spaces, budgets, evaluation indices and seeds. The `data` and `model` blocks
  were compared for the seven recipes at vocabulary 260.
- **Cohort.** The committed indices point into the cohort's validation split, so
  `data.allowed_ids_path` (`${data.cache_dir}/allowed_ids.json`) must be the published
  file. The cohort was not rebuilt here (the dataset was not downloaded); the 256
  stored test stories are all in the published cohort and fit every capacity.
- **One seed per cell.** Gaps between methods are often within half a BLEU point.
- **Two launches, one protocol.** The no-SC cases at 260 to 5000 were launched after
  the others (see `expected.csv` notes). They use the same source, environment, data,
  budget and policies; only `feedback=none` differs.
- **Sampler collapses.** Some EB and re-noising policies score near zero on some
  checkpoints. For example, EB greedy stays below 2.2 validation BLEU on every Gaussian
  model and below 0.5 without SC. Selection is on validation, so a collapsed policy is
  simply not chosen.
- **AR policies.** AR searches only two policies (sampled and greedy decoding), not the
  five diffusion policies, and AR runs need the `validate_inference` change described
  above.
