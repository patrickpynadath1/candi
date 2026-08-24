# Separable two-axis blur implementation

The implementation lives in `algos/blurring_diffusion.py` and is shared by
`algos/candi_blurred.py`. In DCT coordinates its attenuation is

`A(t)[k,j] = A_pos(t)[k] A_feat(t)[j]`.

`feature_rho` splits the common geometric heat time: positional rates receive
`rho`, and declared feature rates receive `1-rho`. Consequently `rho=1` is
exactly the previous positional-only process and `rho=0` is feature-only.
Feature rates are ascending geometric constants; column zero is slowest. They
are not estimated from the embedding covariance and must not be inverted.

The reverse transition uses the same outer-product attenuation at both times,
including the required `A(s)` re-blur on the clean prediction. No score or
epsilon conversion, and therefore no division by `A(t)`, is used.

## Configuration

The main controls in `configs/algo/blurring_diffusion.yaml` are:

- `feature_rho`
- `feature_lambda_min` and `feature_lambda_max`
- `column_init_alpha`
- `detach_forward_embedding`
- `column_norm_log_every`

Sweep `feature_rho` over `0, .25, .5, .75, 1`, retaining `1` as the exact
positional-only reference. Set `column_init_alpha=0` for uniform initialization.

## Constraints and diagnostics

The small input table is projected toward zero column mean after every optimizer
step and normalized to unit row norm whenever it is used. Initialization gives
slow columns larger variance. Column norms are retained in
`column_norm_history`, with first/last norms and their ratio logged to Lightning.

Startup validation checks positional and feature death ordering, crossing-time
bunching, terminal signal leakage, and Markov variance. The first real batch
adds a data-power-aware SNR diagnostic and a terminal normality estimate.
Padding/conditioning masks are converted into independent contiguous spans, so
padding values cannot enter a generated span through the DCT.

## Important gradient-path limitation

In the current architecture, `E_in` is untied from `E_out`, and the training
objective is vocabulary cross-entropy through `E_out`. Corruption is the only
training graph that touches `E_in`. Therefore:

- the default is `detach_forward_embedding=false`, matching the original
  positional-only and CANDI-blurred training behavior;
- `detach_forward_embedding=true` freezes `E_in` apart from hard projection;
- evolving column norms cannot demonstrate learned semantic migration in that
  setting;
- monitor pairwise geometry and column norms in the live default for the
  proposed near-orthogonality cheating mode.

Resolving this while retaining detachment requires an additional explicitly
designed learning objective for `E_in`; none is silently introduced here.

The AR-probe total-correlation measurement, topic probe, full training ablation
matrix, and sampling step-count curves are experiments rather than forward-
process primitives. They remain downstream runs, not startup assertions.
