# CANDI + Blurring + Entropy-Bounded Sampling on Sudoku Hard

## Status

Yes. We trained two full CANDI-blurred models for 20,000 optimizer steps and
evaluated their final checkpoints on all 2,000 examples in the Sudoku-hard
validation set. These were not smoke tests or 128-example callback estimates.

The evaluation used 180 available reverse steps, entropy bound
`gamma = 0.1`, and greedy (`argmax`) clean-token commitments. The model could
finish early once every generated coordinate had been accepted, so the measured
NFE is much smaller than 180.

## Full evaluation results

| Method | Input embedding dim. | Solved | Exact solve / valid-grid rate | Cell accuracy | Mean NFE | NFE range |
|---|---:|---:|---:|---:|---:|---:|
| **CANDI-blurred + EB-0.1 + greedy** | **512** | **1,835 / 2,000** | **91.75%** | **98.644%** | 3.409 | 1–11 |
| **CANDI-blurred + EB-0.1 + greedy** | **16** | **1,823 / 2,000** | **91.15%** | **98.470%** | 4.0535 | 2–14 |
| MDLM + EB-0.1 + greedy | — | 1,809 / 2,000 | 90.45% | 98.489% | 3.1525 | 1–11 |
| CANDI + EB-0.1 + greedy | — | 1,560 / 2,000 | 78.00% | 96.698% | 3.493 | 1–10 |

Additional structural-validity rates were:

| Method | Valid rows | Valid columns | Valid boxes |
|---|---:|---:|---:|
| CANDI-blurred, `d_in=512` | 98.767% | 99.211% | 99.444% |
| CANDI-blurred, `d_in=16` | 99.283% | 98.444% | 99.361% |
| MDLM | 98.889% | 98.822% | 99.339% |
| CANDI | 97.633% | 96.656% | 98.739% |

The raw result files are:

- `new_eval_results/sudoku_hard_eb_0p1_greedy/candi_blurred_d512/results.json`
- `new_eval_results/sudoku_hard_eb_0p1_greedy/candi_blurred_d16/results.json`
- `new_eval_results/sudoku_hard_eb_0p1_greedy/mdlm/results.json`
- `new_eval_results/sudoku_hard_eb_0p1_greedy/candi/results.json`

The CANDI-blurred checkpoints are under
`new_sudoku_runs/candi_blurred_d{16,512}_hard_eb/checkpoints/`. Each run contains
checkpoints at 5k, 10k, 15k, and 20k steps plus `last.ckpt` and `best.ckpt`.

## What we implemented

### Architecture

The method retains CANDI's bidirectional transformer, discrete reveal process,
and likelihood objective, but replaces CANDI's independent continuous rank noise
with a heat-kernel Gaussian corruption over sequence positions.

There are separate, untied embedding tables:

- `E_in in R^(V x d_in)` represents the continuous corrupted state. Its rows are
  normalized whenever they are used.
- `E_out in R^(V x 512)` is the unrestricted classification table used for logits.

The two runs ablate `d_in in {16, 512}` while keeping the transformer hidden width
and output geometry at 512. The input path projects `d_in -> 512`; the output path
projects `512 -> 512` before multiplication by `E_out^T`.

### Heat-kernel forward corruption

Let `X_0 in R^(L x d_in)` be the normalized clean input embeddings for the
generated span and let `V` be the fixed orthonormal DCT-II matrix. For Sudoku,
`L=89`: the 81 solution digits plus eight row separators. The 91-token puzzle
prefix is conditioning and is not part of the DCT span.

```text
V[n,k] = c_k cos(pi (n + 1/2) k / L),
c_0 = sqrt(1/L),  c_k = sqrt(2/L) for k > 0.
```

The positional modes are `Y_0 = V^T X_0`. Their effective decay rates are

```text
lambda_k = lambda_floor + lambda_scale * 2(1 - cos(pi k / L)).
```

For time `t in [0,1]`, the implementation uses a shifted geometric heat time,
which both preserves logarithmic mode spacing and makes the clean endpoint exact:

```text
tau(t) = tau_min exp(t log(tau_max/tau_min)) - tau_min,
a_k(t) = exp(-lambda_k tau(t)),
sigma(t) = t^p,  with p = 2.
```

The corrupted modes and state are

```text
Y_t[k] = a_k(t) Y_0[k] + sigma(t) epsilon_k,
epsilon_k ~ N(0, I),
Z_t = V Y_t.
```

`tau_min`, `tau_max`, and `lambda_floor` are derived from the configured endpoint
attenuations. At the clean endpoint every mode is exactly untouched because
`tau(0)=0`; at the terminal endpoint the lowest nonzero mode has attenuation
0.01 and the mean
mode has terminal attenuation 0.03. Startup checks verify ordered SNR crossings,
limit mode-death bunching, require terminal mean-mode SNR below 0.1, and verify
nonnegative reverse variances.

The DCT operator and its schedule are fixed and data-independent, so gradients do
not change the corruption matrix. We intentionally allow gradients from the
corrupted input into `E_in`; normalization prevents the table from changing its
row norms. Padding and conditioning masks are handled as contiguous spans, so the
puzzle prefix and padded values cannot leak into the blurred solution span.

### Combination with CANDI's discrete state

CANDI first applies its discrete corruption and obtains a reveal mask `m`.
The blurred continuous state is calculated from the clean embedding sequence, but
the actual transformer input is

```text
X_t[i] = E_in[x_0[i]]   if m[i] = 1,
         Z_t[i]         if m[i] = 0.
```

Thus a revealed coordinate is clean in the final model state; only unrevealed
coordinates receive the blurred representation. During sampling, puzzle-prompt
coordinates and entropy-accepted coordinates are absorbing and are restored after
each continuous transition.

There is one important coupling detail: the DCT blur is computed over the complete
89-token solution span before clean revealed coordinates are restored.
Consequently, revealed solution-token content can influence blurred values at
other solution positions through the global DCT modes, although the revealed
coordinates themselves remain clean. The puzzle prefix does not participate in
this transform.

The training loss remains CANDI's time-weighted clean-token cross-entropy. Revealed
positions are forced to their known token, and the transformer predicts factorized
clean-token marginals at unrevealed positions.

### Exact reverse Gaussian transition and re-blurring

At a reverse step from noisier time `t` to cleaner time `s < t`, the network
predicts clean-token probabilities. CANDI-blurred converts the clean prediction to
the small input space (for these greedy runs, by embedding the per-position
`argmax`) and transforms both the state and prediction to DCT modes.

For each mode, define

```text
r_k = a_k(t) / a_k(s),
d_k^2 = sigma(t)^2 - r_k^2 sigma(s)^2.
```

The implemented posterior moments are

```text
Var[Y_s,k | Y_t,k, Yhat_0,k]
  = d_k^2 sigma(s)^2 / sigma(t)^2,

E[Y_s,k | Y_t,k, Yhat_0,k]
  = (sigma(s)^2 / sigma(t)^2) r_k Y_t,k
    + (d_k^2 / sigma(t)^2) a_k(s) Yhat_0,k.
```

The factor `a_k(s)` is the required **re-blur gate**. Although the model predicts a
sharp endpoint, a single reverse step may only inject as much of each positional
mode as should exist at time `s`. No inverse `1/a_k(t)` deconvolution is used.
Sampling draws from this posterior for `s > 0` and takes the posterior mean at the
clean endpoint.

### Entropy-bounded greedy sampler

For every still-active coordinate, the sampler computes predictive entropy

```text
H_i = -sum_v p_i(v) log p_i(v).
```

Coordinates are ordered from lowest to highest entropy. It accepts the largest
prefix `S` satisfying the implementation's excess-entropy budget

```text
sum_(i in S) H_i - max_(i in S) H_i <= gamma,
gamma = 0.1.
```

At accepted positions it commits

```text
xhat_i = argmax_v p_i(v),
```

and replaces the continuous state with the corresponding normalized `E_in` row.
Accepted coordinates then remain fixed. Rejected coordinates are not reset: they
retain the state produced by the current reverse Gaussian transition and continue
along the blur trajectory. A sample terminates as soon as all 89 generated
coordinates (81 digits and eight separators) are accepted, which explains mean
NFEs near 3–4 despite a maximum schedule of 180.

## Discussion

The strongest result is the 512-dimensional blurred model at 91.75%. It exceeds
the matched greedy-EB CANDI result by 13.75 percentage points and the matched MDLM
result by 1.30 points. The 16-dimensional model is only 0.60 points behind the
512-dimensional model, despite compressing the continuous input geometry by 32x.
That supports the idea that the positional blur supplies useful shared structure
without requiring a high-dimensional continuous channel.

The `d_in=512` run also finishes about 16% earlier than `d_in=16` on average
(3.409 versus 4.0535 NFE), suggesting that its predictions become confidently
committable sooner. On the other hand, the small difference in exact solve rate is
not enough by itself to establish a definitive embedding-width effect; paired
bootstrap confidence intervals over the same 2,000 puzzles would be the right next
analysis.

These results establish that the combined method works well under one particular
sampler and one 20k-step training budget. They do **not** yet isolate whether the
gain comes from the heat-kernel corruption, the greedy EB acceptance rule, their
interaction, or training-run variance. The most useful next table would evaluate
the same checkpoints with ancestral CANDI, EB with sampled clean predictions, and
greedy EB across several `gamma` values. A step-count curve would additionally test
whether the re-blurred reverse transition improves gracefully as the available
number of steps is reduced.
