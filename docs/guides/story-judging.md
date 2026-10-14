# TinyStories judge evaluation

Run judging after generation, separately from training and the cheap TinyStories
metrics. This command uses the existing `transformers` dependency; it does not
need the `prometheus-eval` package or a paid API. No judge runs automatically during
training. The default judge is `prometheus-eval/prometheus-7b-v2.0` (Prometheus 2 7B).

The input is a task evaluation JSON object with a `records` list, a JSON array,
or JSONL. Every sample needs a unique nonempty string `id`, `prompt`, and
`continuation`. `reference` is optional. TinyStories task evaluation exports these
fields. Preserve IDs when combining runs; prefix them with run ID if needed.

```json
{"id":"validation:example-1","prompt":"Once there was a cat.","continuation":"The cat found a friend. They played happily.","reference":"The cat went home to sleep."}
```

Inspect prompts without loading a model (CPU-only, no network):

```bash
python -m core.evaluation.story_judge \
  --input results/tinystories-evaluation.json \
  --output results/story-judge-prompts.json \
  --prepare-only
```

Run on a locally cached model or local checkpoint directory:

```bash
python -m core.evaluation.story_judge \
  --input results/tinystories-evaluation.json \
  --output results/story-judge-scores.json \
  --model /absolute/path/to/prometheus-7b-v2.0 \
  --device cuda:0 --batch-size 2
```

Use a free GPU, separate from active training jobs. CPU is the default device but
7B inference on CPU is usually slow. CPU loading uses float32 for compatibility
with CPUs lacking native bfloat16 support; GPU loading uses the checkpoint dtype.
To explicitly permit the first download,
use `--model prometheus-eval/prometheus-7b-v2.0 --allow-download`. Downloads are
otherwise disabled. For reproducible hub loads, set `--revision` to a commit SHA.
The model tokenizer must supply a compatible chat template. Batch size controls
memory use; there are four requests per story. Decoding is greedy, with a default
512-token feedback budget and seed 0. Inputs beyond 7,000 tokens fail visibly
rather than silently truncating the story; adjust `--max-input-tokens` only within
the model's context budget, allowing room for `--max-new-tokens`.

The local backend supplies both a judge system message and the grading request.
Prompt version `story-v2` replaces the original literal `[RESULT] N` instruction,
which real Prometheus inference copied instead of emitting a numeric marker.
Keep this prompt version fixed across comparisons.

An actual GPU smoke test used revision
`66ffb1fc20beebfb60a3964a957d9011723116c5`, batch size 4, greedy decoding and a
512-token feedback limit. All eight judgments on two hand-written samples parsed:
a coherent continuation scored 4/5 for grammar and 5/5 for the other dimensions;
a deliberately broken continuation scored 1/5 throughout. This verifies the
inference and parsing path, not judge calibration or generated-model quality.

## Rubrics and interpretation

Four fixed five-level rubrics grade grammar, coherence, consistency with the
beginning, and nonredundancy. The exact rubrics and full prompts are saved in the
result. These are this project's versioned rubrics, **not a reproduction of a
published TinyStories/ILM grading protocol**. The example reference is explicitly
not the only acceptable continuation. Prompt fields are JSON-quoted and described
as data; as with any LLM judge, this does not guarantee immunity to prompt injection.

The output includes means, scored/failed/total counts per dimension, per-request
raw feedback, integer scores, sample IDs, and explicit errors. Missing or malformed
scores are `null`, never zero. Means use successful responses only, so **always
compare coverage before comparing means**. Any failed request causes a nonzero
exit after saving the results. Provider exceptions and response-count mismatches
fail the affected batch. Input ID errors fail before inference. Existing outputs
are not overwritten.

Metadata records rubric version, exact rubric text, model/revision, resolved hub
commit when available, software versions for local inference, input-file SHA-256,
batching and decoding settings. Pin a model snapshot and environment when making
comparisons; a local directory can change even if its pathname stays the same.
Judge ratings are subjective ordinal scores, not solve rates. Review a sample of
ratings manually and retain average length, repetition, and diversity alongside them.

## Score responses from another provider

The prepare-only output contains `requests`, each with a unique `id` and `prompt`.
Send these prompts to your chosen judge separately, then save JSONL containing
exactly one `{ "id": <request id>, "output": "Feedback ... [RESULT] 4" }` per
request. The tool makes no external API calls. Grade the saved responses with:

```bash
python -m core.evaluation.story_judge \
  --input results/tinystories-evaluation.json \
  --responses results/provider-responses.jsonl \
  --model your-provider-and-exact-model-version \
  --output results/story-judge-scores.json
```

The response IDs must match exactly; file order does not matter. Extra, duplicate,
or missing IDs are rejected. Save the external provider's actual generation
settings alongside its responses: offline mode's CLI settings describe the local
requested protocol, not proof of the provider settings used. Parsing requires one
final `[RESULT]` marker with an integer 1–5. Reassess failed responses explicitly;
there is no silent retry or substitution.

The unit tests use a fake provider and exercise parsing, failure accounting,
batching, sample alignment, and the complete prepare/offline-score CLI. They do
not certify the grading quality or memory usage of a real 7B judge.

Reference: [Prometheus official implementation](https://github.com/prometheus-eval/prometheus-eval).
