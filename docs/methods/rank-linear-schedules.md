# Rank-linear simplex and spherical schedules

These schedules change the time parameterization of existing corruption paths.
Normalized expected rank means the expected fraction of competing tokens that
score above the clean token. It is not the probability of a top-1 decoding error.
With symmetric noise, its endpoints are 0 (clean) and 0.5 (uninformative).

## Simplex derivation

For uniform vocabulary reference and physical progress u, the implemented kernel is
Dirichlet(c(u) * ((1-u) * one_hot(x) + u/V)). The target coordinate has parameter
`a=c(u)*(1-u+u/V)` and a competing coordinate has `b=c(u)*u/V`.
The normalized-Gamma representation gives `z_x/(z_x+z_j) ~ Beta(a,b)`.
Consequently, `P(z_j > z_x) = I_{1/2}(a,b)`, the regularized incomplete Beta CDF.
Log-space storage does not alter coordinate ranks.

`SimplexRankSchedule` evaluates and inverts that curve while retaining the base
concentration schedule. It returns the physical `SimplexPath.time`, so the reverse
bridge uses the same warped physical coordinates as forward corruption.
Nonuniform reference distributions are rejected by hybrid-view validation.
A nonmonotone rank curve is rejected instead of silently sorting the path.

## Spherical derivation

Assume independent, uniformly distributed unit codewords in dimension d, scored
by dot product. Let theta be the angle between a clean codeword and uniform noise.
At physical SLERP progress u, the clean score is cos(u*theta).
For an independent competitor, its dot product C with the noisy state satisfies
`(C+1)/2 ~ Beta((d-1)/2, (d-1)/2)`. Hence

```text
r(u) = E_theta[ I_{(1-cos(u*theta))/2}((d-1)/2, (d-1)/2) ].
```

The angle density is proportional to sin(theta)^(d-2). We integrate using the
existing Gauss-Jacobi sphere-cosine quadrature and invert a monotone lookup table.
In d=2, theta is uniform on [0,pi], giving the independent identity `r(u)=u/2`.

This is exact for the random-codebook ensemble up to numerical integration and
interpolation. It is not calibrated to a particular learned codebook. The existing
spherical representation remains trainable; using it adopts this approximation.
The schedule returns keep=1-u and its derivative, with no auto-truncation of the
original spherical path.

## Configuration and endpoints

Use `hybrid_view@model.views.content=simplex_rank` or
`hybrid_view@model.views.content=spherical_rank` in the corruption-factorial recipe.
Both support the existing self-conditioning and multi-view machinery.

```bash
uv run --locked python main.py +experiment=corruption_factorial \
  hybrid_view@model.views.content=spherical_rank --cfg job
```

Defaults match the Gaussian embedding rank convention: `rank_min=0.01`,
`rank_max=0.49`, `endpoint_width=0.01`. Rank is linear on [0.01,0.99]; short
physical-progress ramps connect to exact clean/noise endpoints. It is therefore
not globally rank-linear in the endpoint ramps. For full-range linear rank, set
`rank_min=0`, `rank_max=0.5`, and `endpoint_width=0`; endpoint slopes then depend
on the finite lookup resolution. Defaults use 8193 grid points and (spherical)
256 quadrature nodes. Tables are module buffers and follow device/checkpoint moves.
AdaptiveTimeMap can wrap either rank base through the existing view time_map.

## Verification

`core/tests/test_rank_linear_paths.py` checks analytic rank inversion, actual
Dirichlet and SLERP corruption simulations with 30,000 samples per point,
quadrature against the circle identity, spherical derivatives against finite
differences, endpoints, and train/backward/generation for both schedules crossed
with masked/uniform views and self-conditioning off/on. Unsupported analytic
assumptions are explicit rather than inferred from training results.

Observed simulation ranks (seeded CPU verification):

| Kernel | Geometry | Target 0.054082 | Target 0.250000 | Target 0.445918 |
| --- | --- | ---: | ---: | ---: |
| Simplex | V=12 | 0.052333 | 0.248567 | 0.445767 |
| Simplex | V=260 | 0.054233 | 0.247000 | 0.448500 |
| Spherical | d=3 | 0.055667 | 0.248500 | 0.446367 |
| Spherical | d=32 | 0.053833 | 0.250067 | 0.446200 |
| Spherical | d=512 | 0.055967 | 0.251967 | 0.446133 |

The separate analytic inversion checks require absolute error below 0.00003;
Monte Carlo tolerances are wider to account for sampling noise.
