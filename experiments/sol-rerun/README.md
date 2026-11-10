# sol-rerun: TinyStories scored with SOL, samplers tuned by successive halving

An inference-only rerun of the TinyStories cases of [continuous-schedules](../continuous-schedules/)
and [hybrid-sc](../hybrid-sc/) on their trained checkpoints. Three things change; the
recipes, data cohort and checkpoints do not.

1. **Score.** [SOL](../../docs/tasks/sol.md) against the real continuations of the same
   prompts (GPT-2 Large final-layer states, 1024 x 1024 slices, lower is better) replaces
   BLEU. BLEU, with and without its brevity penalty, is still recorded.
2. **Samplers.** The five policies of `hybrid_renoising` are kept as they were; EB with
   random uncommitment (`eb_churn`: rate, stop fraction, entropy bound, temperature) is
   added for every case, and the [Ψ-sampler](../../docs/components/psi-sampler.md) (`psi`:
   schedule, η, loop t_on/t_off, top_p, temperature) for the hybrid cases, where it acts
   on the masked or uniform view.
3. **Tuning.** Successive halving: every trial is scored on 64 tuning prompts, the best
   third on 256, the best third of those on 1024, and the top 3 are re-scored on the
   1024 with a fresh seed (2041), with a 200-replicate paired bootstrap interval of SOL.
   Each policy and budget is tuned separately at NFE 128 and 512; each budget's winner is
   scored once on the test split.

## Protocol

`evaluation/tinystories-bpe-10000.json`: from the 13,092-story validation cohort, a
2,048-prompt tuning pool (in draw order; rungs are its prefixes) and a disjoint
4,096-prompt test partition, drawn with `numpy.random.default_rng(20261110)`. Tuning seeds
41 and 42, test seeds 1041 and 1042. The case names end in `-sol` and the outputs go to new
directories, so nothing of the source runs is resumed or overwritten.

| Sweep | Cases | Policies (trials) |
| --- | --- | --- |
| `+sweep=continuous_schedules_sol` | 8: 4 continuous paths x {no SC, SC} | native 16, native_greedy 2, eb_greedy 16, renoising 16, renoising_greedy 2, eb_churn 24 |
| `+sweep=hybrid_sc_sol` | 16: {masked, uniform} x 4 paths x {no SC, SC} | the same plus psi 24 |

The checkpoints are read from `preparation.source_runs`
(default `$SDDC_STORAGE_ROOT/runs/continuous-schedules` and `.../runs/hybrid-sc`), at
`<source case>/training/candidates/44136fa355b3678a1146/training/checkpoints/last.ckpt`
(the single fixed training candidate). `--check` refuses a case whose checkpoint is
missing; prepare on the machine that holds them.

## Run

```bash
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=continuous_schedules_sol preparation.output=$SDDC_STORAGE_ROOT/plans/continuous-schedules-sol
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
  $SDDC_STORAGE_ROOT/plans/continuous-schedules-sol
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/continuous-schedules-sol/manifest.json --check
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/continuous-schedules-sol/manifest.json \
  --output $SDDC_STORAGE_ROOT/runs/continuous-schedules-sol --address auto --case-concurrency 4
```

and the same with `hybrid_sc_sol` / `hybrid-sc-sol`. GPT-2 Large is downloaded from the
Hugging Face Hub at its pinned revision on first scoring (set `HF_HOME` to a disk with
space); reference features are cached under `$SDDC_STORAGE_ROOT/sol_cache`.

## Cost

Generation work from `--check` (sequences x NFE, an upper bound; EB samplers stop early):

| Plan | Sequences per case | Sequence network calls | vs. the source sweep's TinyStories cases |
| --- | --- | --- | --- |
| continuous-schedules (current) | 13,824 | 14.2M (8 cases) | 1x |
| continuous-schedules-sol, NFE 128 and 512 | 132,096 | 338M | 24x |
| continuous-schedules-sol, NFE 128 only | 66,048 | 68M | 4.8x |
| hybrid-sc (current) | 13,824 | 28.3M (16 cases) | 1x |
| hybrid-sc-sol, NFE 128 and 512 | 164,864 | 844M | 30x |
| hybrid-sc-sol, NFE 128 only | 82,432 | 169M | 6.0x |

The 4,096 x 2 test generations per budget are a large share (8,192 of the ~58k/74k
sequences per budget). At the continuous-schedules rate (1.8-5.2 MI210 hours per
TinyStories case for 13,824 sequences at NFE 128), expect roughly 9-25 h per
continuous case at NFE 128 only and 45-125 h with NFE 512 added, plus SOL scoring (one
GPT-2 Large pass per generated story and a second or two of slicing per evaluation).
Pass `preparation.evaluation.nfe=128` when preparing to drop the 512 budget (after a
run has started, any protocol change needs new plan and case names).
