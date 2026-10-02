# OpenWebText

[OpenWebTextTask](../../core/tasks/openwebtext.py) supplies unconditional text
generation with GPT-2 tokenization, a base vocabulary of 50,257, and default
length 1,024. The model's mask token is configured separately.

```bash
uv run python main.py +experiment=openwebtext
uv run python main.py +experiment=openwebtext mode=evaluate \
  evaluation.checkpoint=/path/to/native.ckpt task.num_eval_samples=128
```

Checkpoint architecture and modifiers must match the selected configuration.
See [checkpointing](../guides/checkpointing.md) for legacy import and
[frontier sweeps](../guides/frontier-sweeps.md) for a pretrained MDLM example.

## Training data

The [data config](../../configs/data/openwebtext.yaml) selects
`Skylion007/openwebtext`, with the last 100,000 documents held out. Data may
instead come from `data.files_train` / `data.files_validation` lists of JSON
files with `text` fields, or `data.prepared_train` / `data.prepared_validation`
directories saved by the datasets library with length-matched `input_ids`.
`limit_documents_*` caps preprocessing input; `limit_train` and
`limit_validation` cap packed sequences. `revision` pins a dataset revision.

[Packing](../../core/data/openwebtext.py) appends EOS to each document,
concatenates groups of eight documents, splits into bodies of length L−2, and
wraps BOS/EOS. Each group's incomplete tail is dropped. All tokens participate
in training loss/corruption. Unconditional evaluation uses a count-only dataset
and therefore does not need the training or held-out corpus; it still needs the
generation tokenizer and reference scorer.

## Generation metrics

[CausalReferenceScorer](../../core/tasks/metrics/generative.py) defaults to
GPT-2-large. It decodes generated samples, retokenizes with the reference
tokenizer, truncates to the task length, and computes shifted next-token cross
entropy in nonoverlapping contexts (`task.scorer.context_size=1024`). The first
token of each context has no prediction target. Padding and tokens following
the first EOS in a context are excluded; the EOS target itself is included.

GPT-2 shares its BOS/EOS ID. `task.scorer.ignore_initial_bos=true` permits that
ID at the beginning of the document, so BOS-prefixed samples receive scores.
Setting it false recreates the old boundary behavior, which could score zero
tokens. This correction can change historical perplexities.

Corpus perplexity is `exp(sum NLL / sum scored_tokens)`, **not** the mean of
sample perplexities. No scored tokens yields `null`. This is a reference
language-model plausibility measure, not the generator's likelihood.

Entropy is empirical unigram entropy `−Σ p(v) log p(v)` in nats per generated
sequence, including special tokens; the aggregate averages these sequence
entropies. It is not predictive entropy or pooled-corpus entropy. Reports retain
sample text, entropy, NLL and scored-token count, plus the reference model name.

`task.scorer.batch_size`, `device`, `local_files_only`, and `model_name` control
the independent scorer. Its precision does not inherit generator BF16 autocast.
Perplexity and entropy should be read together: either metric alone can reward
undesirable repetition or incoherent diversity.

A real 65,000-step pretrained MDLM EMA checkpoint was checked on 32 samples:
perplexity 30.7164, entropy 4.92239 nats, temperature .8 and 33 NFE. This verifies
execution at one point, not an established frontier. See
[reproduction results](../validation/reproduction-results.md).
