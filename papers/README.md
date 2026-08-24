# Paper-to-algorithm pipeline

This directory is the intake and design record for methods implemented in Duo.
The goal is to make a paper implementation reviewable before a long training run
makes mistakes expensive.

## Directory layout

```text
papers/
├── inbox/                 # PDFs waiting to be scoped
├── methods/<slug>/
│   └── implementation.md  # Living source-to-code record
└── templates/
    └── method-plan.md
```

PDFs stay in `papers/inbox/`; the method record links to the source PDF and
stores its SHA-256 digest. Use a short, stable, lowercase slug such as
`replaid` or `discrete-flow-matching`.

## Start a method

1. Put one PDF in `papers/inbox/`.
2. Scaffold its implementation record:

   ```bash
   uv run python scripts/new_paper_method.py \
     papers/inbox/2605.18530v1.pdf \
     --slug replaid \
     --title "Continuous Diffusion Scales Competitively with Discrete Diffusion for Language" \
     --paper-id arXiv:2605.18530v1
   ```

3. Read the paper in two passes. First fill in the claim, dependencies, and
   acceptance criteria. Then transcribe every implementation-relevant equation,
   algorithm, and appendix detail into the record with page/section citations.
4. Resolve every item under **Open questions** before changing the training
   objective. If the paper is ambiguous, record the chosen interpretation and a
   test that would expose a wrong choice.

The scaffolder rejects non-PDF inputs, refuses to overwrite a method record,
and records the exact source checksum. Re-running it after replacing a PDF is a
deliberate error rather than silently changing the source under an implementation.

## Implementation contract for this repository

Map the method into these surfaces; omit a surface only with a reason in the
record.

| Concern | Repository location | Required evidence |
| --- | --- | --- |
| Training objective and transitions | `algos/<slug>.py` | equation-level unit tests, finite loss, backward pass |
| Algorithm registration | `@algo_registry.register("<name>")` | `tests/unit/algos/test_registry.py` |
| Model architecture | `models/<name>.py` and `models/registry.py` | shape, masking, conditioning, and gradient tests |
| Sampling trajectory | `samplers/<name>.py` and `samplers/registry.py` | declared family/capabilities and sampler contract tests |
| Default method config | `configs/algo/<name>.yaml` | real Hydra composition/constructor test |
| Reproducible experiment | `configs/experiment/<name>_<task>.yaml` | smoke command and expected artifact/metric |

Algorithm modules are auto-imported by `algos/__init__.py`; do not add a manual
import. Sampling loops belong in `samplers/`, not in the trainer. New samplers
must follow [the sampler contract](../samplers/README.md).

## Definition of done

A method can move from `scoped` to `implemented` only when:

- the record maps paper equations to named code and tests;
- the shipping Hydra config resolves a registered algorithm and backbone;
- a tiny real model produces a finite loss and non-zero finite gradients;
- sampler compatibility is validated before generation;
- edge cases and numerically sensitive expressions have focused unit tests;
- a CPU or smallest-practical smoke run and its exact command are recorded;
- deviations from the paper and unverified claims are explicit.

Run focused tests first, then the integration gates:

```bash
uv run pytest tests/unit/algos/test_<name>.py -q
uv run pytest tests/unit/samplers/test_<name>.py -q  # when sampling changes
uv run pytest tests/unit/algos/test_registry.py -q
uv run pytest tests/unit/algos/test_real_configs.py -q
uv run ruff check algos models samplers tests scripts/new_paper_method.py
```

Large-scale reproduction is a separate `reproduced` milestone. An implementation
must not claim reproduction solely because it trains or matches tensor shapes.

