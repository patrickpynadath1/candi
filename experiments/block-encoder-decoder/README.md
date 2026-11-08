# Block diffusion: scale the encoder or the decoder?

Block diffusion writes a sequence one block at a time: each block is denoised while the
model reads the clean blocks before it. A shared block transformer (BD3-LM) does both jobs
with one network. E2D2 splits them: an encoder reads the clean context once per block, and
a decoder denoises the current block at every step. So the two have different costs, and
the question is which one is worth making deeper.
([blog section](https://patrickpynadath1.github.io/blog/diffusion-lm-design-choices),
"Block Diffusion: Scale the Encoder or the Decoder?")

## Protocol

One protocol for all 27 cases: the one the published (AMD MI210) runs used.

**Models** (`configs/block/`, one option per model), each at width 512, 8 heads:

| `block=` | Model | Encoder layers | Decoder layers |
| --- | --- | ---: | ---: |
| `encoder1_decoder8`, `encoder2_decoder8`, `encoder4_decoder8` | E2D2, encoder arm | 1, 2, 4 | 8 |
| `encoder8_decoder1`, `encoder8_decoder2`, `encoder8_decoder4` | E2D2, decoder arm | 8 | 1, 2, 4 |
| `encoder8_decoder8` | E2D2, both arms' 8/8 point (one model) | 8 | 8 |
| `raw_context` | no encoder: the decoder attends to the clean context's token embeddings | 0 | 8 |
| `shared_transformer` | one shared transformer (BD3-LM) | shared | 8 |

E2D2 and raw context use `EncoderDecoderModel` (see
[encoder/decoder wrappers](../../docs/guides/encoder-decoder.md)); raw context is its
encoder depth 0. The shared model sets `encoder_decoder: null`.

**Tasks:** text8 (256 characters), TinyStories with the 10k BPE tokenizer
(`artifacts/tokenizers/tinystories-bpe`, 512 tokens, a 64-byte story prompt) and
Sudoku-hard (180 tokens, the first 90 are the puzzle). 9 models x 3 tasks = 27 cases.

**Training** (`configs/sweep/block_encoder_decoder.yaml`): masked block diffusion (MDLM
loss, [block diffusion](../../docs/modifiers/block-diffusion.md) with block size 32,
clean prefix, no clean-token loss), time-conditioned AdaLN (`algorithm.time_conditioning=
true`; the clean-context encoder is not time-conditioned), no self-conditioning. 20k
updates of 256 sequences (micro-batch 8 x 32 accumulation steps), constant learning rate
3e-4 after 2.5k warmup steps, EMA 0.9999, bf16, `torch.compile`, activation
checkpointing. One training seed (1).

**Evaluation** (`evaluation/<task>.json`, the published runs' partitions): samplers are
tuned on 128 validation examples with seeds 41 and 42, and only the policy with the best
validation score is run on 256 disjoint test examples with seeds 1041 and 1042. The
score is the mean over the test seeds. Scores: text8, unique valid words of 5+ letters
per sample; TinyStories, corpus BLEU of the continuation against the real one; Sudoku,
fraction of fully correct grids. Budget 128 NFE per sequence, split evenly over the
blocks.

**Samplers:** the native ancestral sampler only: `native` (temperature 0.5 to 1.5, 16
trials) and `native_greedy` (greedy clean estimate; greedy or ancestral last step, 2
trials). Renoising and entropy-bounded samplers were not run on these models.

## Running it

The data first. The published runs used two inputs that are too big to commit, so
`scripts/prepare_data.py` rebuilds them and checks their SHA-256 against the files those runs
used. Download the parquet files yourself
(`huggingface-cli download --repo-type dataset NAME --revision REV`):

```bash
# text8 from the afmck/text8 parquet copy (revision 58c74e96...), as the task's split files.
uv run --locked python scripts/prepare_data.py text8 \
  --parquet PATH/afmck-text8/data --raw-dir $SDDC_STORAGE_ROOT/data_cache/text8/raw
# The TinyStories story cohort the evaluation indices refer to (revision f54c09fd...).
uv run --locked python scripts/prepare_data.py tinystories \
  --parquet PATH/roneneldan-TinyStories/data \
  --output $SDDC_STORAGE_ROOT/data_cache/tinystories_bpe/allowed_ids.json
export TINYSTORIES_DATA_DIR=PATH/roneneldan-TinyStories/data
```

Sudoku-hard is generated on first use (seed 42); set `data.sudoku_num_workers` to speed
that up. Then the usual three steps:

```bash
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=block_encoder_decoder preparation.output=$SDDC_STORAGE_ROOT/plans/block-encoder-decoder
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
  $SDDC_STORAGE_ROOT/plans/block-encoder-decoder        # 27 cases
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/block-encoder-decoder/manifest.json \
  --output $SDDC_STORAGE_ROOT/runs/block-encoder-decoder --address auto --case-concurrency 4
uv run --locked --extra scoring python experiments/block-encoder-decoder/analyze.py \
  $SDDC_STORAGE_ROOT/runs/block-encoder-decoder --output results/
```

One model on one task: add `task=battery/text8 block=encoder1_decoder8` to the prepare
command.

## Cost

About 233 GPU hours of training on one AMD MI210 per case: 6 to 14 hours per run
(text8 69 h, TinyStories 97 h, Sudoku 67 h in total), more for deeper models. Sampler
tuning and the test evaluation add about 11 hours over the 27 cases (244 hours from case
start to final result). Per-run hours are in `expected.csv`.

## Results

Published numbers (`expected.csv`, the website's `block_scaling.json`); test scores, mean
of two test seeds, one training seed. TinyStories: BLEU, and BLEU without the brevity
penalty in parentheses. Eval s: seconds to generate the 256 test examples once.

| Model | Encoder | Decoder | text8 | TinyStories BLEU (no BP) | Sudoku | Train h (text8 / TS / Sudoku) | Eval s (text8 / TS / Sudoku) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E2D2 | 1 | 8 | 10.53 | 6.96 (8.73) | 0.025 | 7.5 / 10.3 / 7.4 | 43 / 41 / 21 |
| E2D2 | 2 | 8 | 10.27 | 6.88 (8.79) | 0.021 | 8.0 / 10.8 / 7.7 | 42 / 40 / 21 |
| E2D2 | 4 | 8 | 10.11 | 6.98 (8.64) | 0.033 | 8.8 / 11.9 / 8.6 | 42 / 42 / 22 |
| E2D2 | 8 | 8 | 9.80 | 6.98 (8.78) | 0.049 | 10.3 / 13.7 / 10.2 | 43 / 44 / 22 |
| E2D2 | 8 | 1 | 5.55 | 6.16 (7.72) | 0.002 | 6.0 / 8.2 / 5.7 | 25 / 18 / 9 |
| E2D2 | 8 | 2 | 7.83 | 6.77 (8.77) | 0.012 | 6.7 / 9.0 / 6.5 | 28 / 22 / 11 |
| E2D2 | 8 | 4 | 9.07 | 6.90 (8.63) | 0.029 | 8.0 / 10.6 / 7.7 | 34 / 29 / 15 |
| raw context | none | 8 | 10.30 | 6.53 (7.80) | 0.021 | 7.1 / 9.9 / 6.8 | 42 / 40 / 21 |
| shared | shared | 8 | 9.93 | 6.90 (7.89) | 0.035 | 6.6 / 12.9 / 6.2 | 46 / 44 / 23 |

Decoder depth matters (text8 5.6 to 9.8 from 1 to 8 decoder layers); encoder depth does
not (1 to 8 encoder layers are within seed noise, and on text8 no encoder scores the same).
Inference time grows with decoder depth, not encoder depth, because the encoder runs
once per block. The shared transformer matches E2D2 8/8 by BLEU but writes longer
TinyStories continuations (88% of the reference length, against 79 to 82% for E2D2), so
without the brevity penalty E2D2 8/8 is ahead.

### How `analyze.py` gets each number

It reads every case under the run roots (`core.analysis`), keeps the block diffusion
cases, and reads the model from the recorded config (`model.n_blocks`,
`encoder_decoder.encoder_layers`, or `encoder_decoder: null` for the shared model).
`score`/`seeds` are the selected policy's test score; `validation` is its validation
score; `nfe` and `eval_seconds` are the means over the test seeds of `mean_nfe` and
`evaluation_seconds` in `final/seed-*/results.json`. `train_hours` is the mtime of the
candidate's `trained.json` minus the mtime of the case's `protocol.json`: wall time from
case start to the trained checkpoint, including data preparation and compilation, as the
published values were computed. TinyStories' `bp`, `len_ratio` and `score_nobp` come from
the stored generations (sacrebleu). On the stored runs it reproduces all 27 published rows
exactly, every field.

## Caveats

- **Sudoku: the block models barely learned the task.** At most 5% of grids are solved,
  where the full-sequence masked model of the same campaign solves 97% (with an
  entropy-bounded sampler these block models were not tuned with). The block settings
  were not validated for Sudoku, and the post makes no claim from these cells. Sudoku generations also used
  84 NFE, not 128: the first two of six blocks are all puzzle and need no denoising.
- **text8 source.** The published runs used the Hugging Face `afmck/text8` parquet copy
  (revision `58c74e966ccc66eab2de6b52fad5b14ddb259fa4`), written out as the 90M/5M/5M
  character split files the text8 task reads, because the original `text8.zip` was not
  reachable from their cluster. This example uses that copy (`prepare_data.py text8`) so
  the evaluation indices pick the same validation windows. The task can also split
  `text8.zip` itself (same 90M/5M/5M rule); whether that gives byte-identical files was
  not checked. `prepare_data.py text8 --raw-dir DIR` compares any split files with the
  published hashes.
- **TinyStories cohort.** The evaluation indices refer to a story cohort, not the full
  validation split: stories that fit every tokenizer of the
  [tokenizer ladder](../tokenizer-ladder/) (vocabularies 260 to 10k) at its capacity, with
  validation stories that also appear in train removed (1,444,319 train and 13,092
  validation stories). `prepare_data.py tinystories` rebuilds it with that rule; the
  sweep reads it from `${data.cache_dir}/allowed_ids.json` and fails if it is missing.
  Its hash check was not exercised here (no copy of the parquet files).
- **One training seed, two test seeds.** The two test seeds of one model differ by up to
  0.75 on text8 and 0.53 BLEU on TinyStories, so smaller gaps between models mean
  little. Only native samplers were tuned.
- **Different code and hardware.** The published runs used commit aa4ee07 with ROCm torch
  2.9.1 on MI210s. The MDLM and block-diffusion code has since moved onto the shared
  `ModularDiT` path (the migration tests show MDLM trains and samples the same there).
  Every setting the sweep records matches the published cases (checked case by case),
  but expect rerun numbers to be close, not equal.
  Case names changed too (`e2d2-encoder-1` is now `e2d2-encoder-1-decoder-8`, and
  `block-shared-8` is `shared-transformer-8`); `analyze.py` reads the architecture, not
  the name.
- **Train hours are approximate.** They come from file timestamps. `protocol.json` is
  rewritten when a case resumes; if it is then newer than `trained.json`, `analyze.py`
  starts the clock at the candidate's `training/config.yaml`, which still overstates a
  resumed run that trained in several sessions.
