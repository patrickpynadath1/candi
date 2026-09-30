# Random grammar pilot

This experiment generates a frozen random grammar over latent core occurrences,
then inserts HubPaths bridges. Training sees a neutral BOS, unmarked node tokens,
and PAD. Roles, features, matching positions, topics, and parse spans are only in
`*_parse.jsonl` sidecars. It is a bounded feature-augmented probabilistic grammar,
not a claim to reproduce natural-language syntax or semantics.

## Grammar family

For each hidden category `a`, generate three random body alternatives, each a
list of one to three child categories, with frozen Dirichlet rule probabilities:

```text
S_a -> OPEN_a[k] BODY CLOSE_a[k]
BODY -> child_1 ... child_m
child_b -> ATOM_b | S_b
```

Children recurse with probability 0.55, up to depth three. Each new span samples
one of four agreement features `k`; its opening and closing tokens share that
feature. Roles have overlapping random lexical support and frozen emission
weights. Feature classes partition the vocabulary, making the intended agreement
independently checkable from the token identities at annotated positions.

There are six categories and at most 40 core occurrences per document. The
categories, alternatives, lexical memberships, emissions, and node features are
seeded. The recursive/agreeing rule *family* is deliberately designed; the
generator does not invent arbitrary grammar formalisms. Different alternatives
can coincide. `GrammarConfig` exposes depth, size, category, feature, and
recursion controls for further experiments.

Lexical emission weights multiply the existing unequal baseline, soft bridge
avoidance, low-rank pair correlations, topic preference, and distance law.
Impossible lexical realizations and over-budget derivations trigger whole-document
rejection, which is counted in metadata. Each bridge has an eight-edge budget.
Core counts and visible lengths vary; the model array width is 314.

## Experiment and control

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m tasks.hubpaths.grammar \
  --output data_cache/hubpaths/grammar_pilot --train 1000 --eval 250 --seeds 0 1 2
```

Each of three seeds produces two corpora: agreeing grammar and broken agreement.
The control independently resamples the closing feature; it retains the same
frozen rules, emission tables, graph, core prior, and sampling settings. Actual
documents are independent samples, not exactly matched trees or token streams.
Each corpus has 1,000 train, 250 validation, and 250 test documents. Finished
sequences cannot repeat across splits. Grammar combinations can recur.

Saved artifacts include `grammar.json`, `graph.npz`, `core_prior.npz`, ordinary
model arrays, parse sidecars, and a combined `report.json`. The pilot keeps parsed
documents in memory for analysis; it is intended for small experiments. The
report includes realized core/visible frequencies, exact unique 1–5-grams,
consecutive-core plug-in MI, distances, conditional bridge entropy, rule usage,
depth/length distributions, and rejection counts. MI is sample-biased; bridge
entropy is conditional on the endpoints, not the marginal observed entropy.

Recompute diagnostics on existing corpora without regenerating:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m tasks.hubpaths.grammar \
  --output data_cache/hubpaths/grammar_pilot --analyze-only
```

## What the probes establish

Multinomial logistic regression predicts the four-way closing feature from
visible-token counts in the last 4, 16, 64, or all prefix tokens. Its regularization
is selected on validation data and accuracy/log loss are measured on held-out
documents. Parse annotations locate query positions and labels; they are not
input features. These are supervised probes at known closing positions, not
end-to-end language-model or parser evaluations. Token-count features discard
word order and cannot demonstrate that a model learned a stack or parse tree.

Two query cutoffs are reported:

* `prefix_probes` stop immediately before the closing core. The final bridge can
  already reveal the target's feature through local graph structure.
* `before_bridge_probes` stop at the previous core, before any interior tokens of
  the bridge into the closer. This isolates the earlier-prefix signal more cleanly.

The broken-agreement condition supplies the empirical control; nominal balanced
chance is 25%, and each result also records a training-majority baseline. Multiple
queries within a document are dependent, so query counts are not independent
sample sizes for confidence intervals. Three grammar/prior seeds assess a small
amount of variation on the same graph; they do not establish broad generality.

## Training integration

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python main.py +experiment=hubpaths_grammar_ar \
  strategy=single_gpu trainer.devices=1
```

This configuration uses the agreed 512-wide, eight-block, eight-head architecture.
The 200-node pilot vocabulary gives about 25.4M total parameters. A two-step GPU
integration check completed; it is not a convergence experiment. Parse data is
excluded from the loader, and conditional-waypoint evaluation rejects this mode.
Use the experiment's report rather than the fixed-core `diagnose` command.
