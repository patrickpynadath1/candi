# Rank-calibrated spherical interpolation

This is a derivation for the SLERP path used by
[Language Modeling with Hyperspherical Flows](https://arxiv.org/html/2605.11125v1),
using the normalized rank-degradation convention from
[CANDI](https://arxiv.org/abs/2510.22510). The rank calibration below is our
extension, not a claimed equation from either paper.

## Exact iid-codebook result

Let e_0,...,e_{V-1} be iid uniform unit vectors in R^d, d >= 2, and let n be an
independent uniform unit vector. Define u in [0,1] as the clean-to-noise SLERP
fraction. Set theta = arccos(e_0 dot n), and write

    z(u) = cos(u theta) e_0 + sin(u theta) v,

where v is a uniform unit vector in the tangent space at e_0, independent of
theta. The correct token's cosine score is cos(u theta).

Normalized rank degradation is

    R(z,e_0) = sum_{j != 0} 1[e_j dot z > e_0 dot z] / (V-1).

The usual ordinal rank is 1 + (V-1) R. Distances and cosine scores induce the
same ordering for unit embeddings. This is geometric input rank, not model-logit
rank or token error probability.

A random vector's angle A to any fixed direction has density

    f_d(a) = Gamma(d/2) / (sqrt(pi) Gamma((d-1)/2)) * sin(a)^(d-2),  0 <= a <= pi.

Its CDF can be written with the regularized incomplete beta function:

    F_d(a) = I_{sin(a/2)^2}(k,k),  k=(d-1)/2.

Conditional on theta and z, a competitor wins iff A < u theta. Its angle has
the same distribution independently of theta. Linearity of expectation gives

    r_d(u) = E[R] = integral_0^pi F_d(u theta) f_d(theta) dtheta
           = P[A < u B],  A,B iid with density f_d.

This is exact for the iid uniform codebook, averaged over codebook and noise.
It is independent of vocabulary size because rank is normalized by V-1. No
independence assumption about rank indicators is needed for their mean.

For 0 < u < 1,

    r_d'(u) = integral_0^pi theta f_d(u theta) f_d(theta) dtheta > 0.

Thus r_d has a unique inverse on [0,.5]. Its endpoints are r_d(0)=0 and
r_d(1)=.5. The latter follows from exchangeability of A and B.

Sanity checks:

- d=2: f_d is uniform, hence r_2(u)=u/2 exactly.
- d=3: r_3(u)=1/2-(1+cos(pi u))/(4(1-u^2)), with the continuous limit 1/2 at u=1.

## High-dimensional approximation

For large d, a random spherical angle is approximately N(pi/2,1/d). Keeping
B random as well as A gives

    r_d(u) ~= Phi(-pi sqrt(d) (1-u) / (2 sqrt(1+u^2))).

This approximation is useful for fixed ranks away from zero in large dimension;
it is not a uniform approximation to the entire clean-endpoint tail.

For target rank r, define

    a = -2 Phi^{-1}(r) / (pi sqrt(d)).

When 0 <= a <= 1, the inverse approximation is

    u ~= tan(pi/4 - asin(a/sqrt(2))).

Near u=1 this becomes

    u ~= 1 + 2 sqrt(2) Phi^{-1}(r)/(pi sqrt(d)).

Replacing the endpoint angle B by exactly pi/2 misses its fluctuations. Near
u=1 those fluctuations double the variance of A-uB, so this shortcut loses a
sqrt(2) factor. The angular-normal approximation above includes both sources.

## Coupling and endpoints

For discrete corruption rho, the geometric rank coupling previously discussed is

    r_target(rho) = .01 * 49^rho,
    u(rho) = inverse_r_d(r_target(rho)).

It calibrates the rank among continuously corrupted positions. If hybrid
corruption preserves a position with probability 1-rho, its mean rank averaged
over both preserved and corrupted positions is rho*r_target(rho).

A nonzero minimum rank truncates the clean endpoint, and rank .49 truncates the
noise endpoint. An independent uniform spherical prior has expected rank .5,
not .49. Preserve these truncations explicitly, or append boundary segments to
reach rank 0 and .5. Do not label a .49 endpoint an exact independent prior.

For an alternative exponential curve with exact endpoints, use

    r_target(rho) = .5 * expm1(kappa*rho)/expm1(kappa),  kappa > 0,

and apply the same inverse. This changes the endpoint convention of the proposed
.01-to-.49 ablation.

Adaptive loss linearization remains a separate one-dimensional clock:

    sampled t -> rho=g(t) -> r_target(rho) -> u -> SLERP.

Geometric rank growth does not itself guarantee linear cross-entropy growth.

## Numerical verification (d=512)

Gauss-Jacobi integration of the cosine density, followed by scalar root finding:

| Target rank | SLERP u | Angular-normal inverse |
|---|---:|---:|
| .01 | .911375230 | .911441270 |
| .05 | .936532693 | .936594244 |
| .10 | .950207824 | .950260652 |
| .25 | .973484916 | .973515899 |
| .49 | .999001819 | .999003037 |

A separate direct-vector SLERP Monte Carlo check with 100,000 draws (seed 2026)
produced .00977, .10077, .49306 for targets .01, .10, .49 respectively, consistent
with sampling error. In this coordinate, informative rank changes occur near
the noise endpoint; u is fraction of the full random arc, not residual signal.

## Fixed learned embeddings

The iid law is a calibration model, not guaranteed by unit normalization. For a
fixed distinct embedding pair with cosine c, a useful exact conditional formula
is available. Let T_m(q)=P[W>q], where W is one coordinate of a uniform vector on
S^(m-1), and extend it by 1 for q<=-1 and 0 for q>=1. For d>=3 and -1<c<1,

    P[competitor wins | theta,c]
      = T_{d-1}(sqrt((1-c)/(1+c)) * cot(u theta)).

This follows by writing the competitor as c e_0 + sqrt(1-c^2) w and projecting
onto the uniform tangent direction v. Integrate over theta with density f_d and
average over competitor pairs. This gives the exact mean rank for a fixed table
without sorting sampled full-vocabulary scores. For large vocabularies, sample
pairs; for Sudoku, all pairs are cheap. Coincident embeddings require an explicit
tie convention; antipodal pairs can be handled directly from their scores.

## Implementation implication

Use a distinct SLERP rank calibrator, tabulate r_d and its inverse, and store the
dimension and table in checkpoints. Existing SphereRankSchedule in
algos/rank_schedules.py calibrates linear interpolation with unnormalized Gaussian
noise; its formula must not be reused for this spherical path. The separate calibrator is now implemented as `SLERPRankSchedule`, with
`algo=candi_slerp` providing the coupled hybrid and pure variants.
