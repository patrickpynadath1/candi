# Predictor parameterization

Posterior and plug-in parameterizations own optional `softcap`, using
`c * tanh(logits / c)` for positive finite `c`. The default `null` leaves raw
model logits unchanged. Set `parameterization.softcap=30` to enable it.

Softcapping defines the model output during both training and inference, matching
the original FLM formula. It happens before posterior constraints, normalization,
and inference-only temperature/top-k/top-p. Applying it before exclusions ensures
that a forbidden token stays forbidden: an excluded negative-infinite logit must
not be turned into a finite capped value. Observed-token constraints still hold.

Training and validation use `training_posterior` or `training_plugin`, which never
apply inference-only shaping. Sampling uses the configured temperature and filters
after softcapping. Explicit greedy decoding shares this ordering. Entropy remains
computed before commitment projection. The absorbing score parameterization is
unchanged; it is not a normalized token distribution.
