# Insertion training and sampling

The variable-length modifier owns deletion, gap supervision, and sequence edits.
Its inner recipe still owns the token representation, corruption, and objective.
Birth laws remain view-specific: masked tokens are born masked; uniform tokens
are always born uniform noise; continuous views can corrupt a gap-token estimate.

## Finite insertion counts

For survival probability S(t), let H = log((1-S(t))/(1-S(s))) for a reverse
step t to s. The default `count_integration: finite` samples a Poisson count
with mean `count_scale * predicted_remaining_count * (1-exp(-H))`.
This is the correct conditional birth **mean** when the predicted remaining
count is exact. It is still a Poisson approximation, not the exact binomial
transition conditional on an integer number of remaining tokens.

For linear survival S(t)=1-t, the fraction is 1-s/t. The old implementation
used H=log(t/s), which overestimates finite-step births, especially near the
endpoint. To replay saved runs using the historical sampler, explicitly set
`sampler.count_integration=integrated_hazard`. Existing result files have not
been rewritten; configs that omit this field now use the corrected default.

## Independent sampling choices

`sampler=insertion` retains native content transitions. Its `tokens.mode` is
`posterior` by default; setting it to `greedy` or `sample` supplies a selected
one-hot clean estimate to those transitions. `final_denoise` controls only the
terminal readout and is intentionally independent.

`sampler=insertion_entropy_bounded` uses the same `TokenSelection` and
`EntropyPositions` components as the fixed-length composed sampler, with greedy
clean tokens and frozen commitments. Accepted tokens, prompt protection, and
self-conditioning feedback are transported together through sequence edits.
Inserted positions are initially uncommitted. Counts remain independently
sampled Poisson variables; greedy token selection does not make lengths greedy.

This preset ranks positions globally. It does not implement FlexMDM's windowed
sampling, so it should not be described as an exact reproduction of its adaptive
sampler. Position policies can also be reused with native transitions without
freezing commitments.

## FlexMDM-style objective

`experiment=flexmdm_style` composes existing CE and gap-count losses, with no
extra gap-token head by default. `KeepRateHazardWeighting` multiplies CE by
-alpha'(t)/(1-alpha(t)), using an explicitly configured keep-rate schedule.
It is independent of the corruption and structural modifier. The schedule must
return derivatives with respect to global training time; a nonidentity view
clock needs a correspondingly composed schedule and chain-rule derivative.

For a token born at b, its conditional mask probability at t is
(1-alpha(t))/(1-alpha(b)); its reverse unmasking hazard is therefore
-alpha'(t)/(1-alpha(t)): the birth-time factor cancels. The preset uses the
linear keep schedule and identity clock, giving token hazard 1/t. Count loss
uses the deletion schedule's own hazard, also 1/t for its default linear path.
Already-visible tokens have zero token CE through the constrained observed
posterior. Loss normalization remains our target-token normalization.

The gap-token auxiliary remains independently configurable via
`variable_length.predict_gap_tokens` and
`variable_length.objective.content_weight`. Enable both to study it; it is not
required by masked or uniform birth laws.

Reference: [FlexMDM training loss](https://github.com/brianlck/FlexMDM/blob/99eb7b7ff9fc4041077ba29a6f652dccaa52ff6b/lightning_modules/any_order.py#L70)
and [interpolant hazards](https://github.com/brianlck/FlexMDM/blob/99eb7b7ff9fc4041077ba29a6f652dccaa52ff6b/interpolant.py#L178).
Correcting these differences is not evidence of improved trained-model quality;
that requires a controlled evaluation of the affected checkpoints or retraining.
