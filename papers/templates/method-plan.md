# {{TITLE}}

- Status: `triage` (`triage` → `scoped` → `implementing` → `implemented` → `reproduced`)
- Slug: `{{SLUG}}`
- Paper ID: {{PAPER_ID}}
- Source: [{{PDF_NAME}}]({{PDF_LINK}})
- Source SHA-256: `{{SHA256}}`
- Added: {{DATE}}
- Owners: unassigned

## Claim and scope

### One-sentence claim

TODO

### In scope

- TODO

### Out of scope

- TODO

### Dependencies and closest existing implementation

- Paper/code dependencies: TODO
- Closest Duo algorithm: TODO
- Reusable components: TODO

## Paper specification

For every implementation-relevant item, cite a section, equation/algorithm, and
PDF page. Use symbols from the paper and state tensor shapes and conventions.

| Component | Paper reference | Executable specification |
| --- | --- | --- |
| Data representation | TODO | TODO |
| Forward/corruption process | TODO | TODO |
| Model prediction/parameterization | TODO | TODO |
| Training objective | TODO | TODO |
| Noise/time schedule | TODO | TODO |
| Reverse process/sampler | TODO | TODO |
| Architecture changes | TODO | TODO |

## Source-to-code map

| Paper item | Planned code | Verification |
| --- | --- | --- |
| TODO | `algos/<slug>.py` | focused equation test |
| TODO | `configs/algo/<slug>.yaml` | Hydra composition test |
| TODO | `samplers/<slug>.py` or existing sampler | sampler contract test |

## Acceptance criteria

### Mathematical/unit

- [ ] TODO: limiting case or analytic identity
- [ ] TODO: shape, masking, and reduction behavior
- [ ] TODO: numerical stability at schedule endpoints

### Integration

- [ ] Shipping config resolves registered algorithm and backbone
- [ ] Tiny real model returns finite loss and non-zero finite gradients
- [ ] Sampler validates the model capability contract
- [ ] Checkpoint save/load works if state structure changes

### Reproduction

- [ ] Dataset/tokenization and preprocessing match the paper
- [ ] Training budget and evaluation metric are recorded
- [ ] Baseline and target result include uncertainty/tolerance
- [ ] Exact command, config, seed, commit, and artifact path are recorded

## Experiment ladder

1. Analytic/unit test: TODO
2. Tiny overfit or CPU smoke test: TODO
3. Small comparison against closest baseline: TODO
4. Paper-scale reproduction (optional and separately reported): TODO

## Deviations from the paper

| Deviation | Reason | Expected effect | How tested |
| --- | --- | --- | --- |
| None yet | | | |

## Open questions and decisions

- [ ] TODO

## Commands and results

```bash
# Add exact commands as work proceeds.
```

| Date | Commit | Config/seed | Result or artifact | Interpretation |
| --- | --- | --- | --- | --- |
| | | | | |

