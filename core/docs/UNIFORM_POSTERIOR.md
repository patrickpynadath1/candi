# Posterior uniform diffusion

Use `+experiment=uniform_posterior` for uniform replacement corruption, an
unconstrained clean-token posterior, cross entropy, and uniform ancestral
sampling. The existing `+experiment=uniform_diffusion` retains its original
plug-in parameterization and objective. Model architecture, noise schedule,
time sampler, task, and self-conditioning remain independently configurable.

Uniform replacement can draw the original token. Unlike absorbing masking,
an observed token therefore does not reveal its clean value: the posterior
must not clamp observed tokens to delta distributions.

## Reverse transition

For vocabulary size V, keep rates a at the noisy time and b at the earlier
(cleaner) time, and observed token k, the transition is

    q(x_s=j | x_t=k, x_0=i)
      = [a/b * 1(j=k) + (1-a/b)/V]
        * [b * 1(j=i) + (1-b)/V]
        / [a * 1(i=k) + (1-a)/V].

`PosteriorUniformProcess` averages these normalized bridges over the predicted
posterior p(x_0=i | noisy views). Its implementation uses O(V) work, without
constructing a V-by-V matrix. At b=1 it returns the predicted clean posterior;
at identical times it returns the observed token, including both endpoints.
The a=0 uniform-noise endpoint is supported.

The old plug-in transition instead forms a marginal from predicted clean
weights and then conditions that marginal on the observed token. These are
different operations: plugging an already conditioned posterior into that
formula would condition on the observation again.

## Composition

For a `core.hybrid.branches.DiscreteBranch`, choose
`core.processes.uniform_posterior.PosteriorUniformProcess` as its process,
with the desired keep-rate schedule and token embedding adapter. The existing
branch update then consumes the shared clean posterior directly. No uniform
sampling mathematics belongs in the hybrid algorithm. Hybrid assembly rejects
plug-in processes with an informative error.

The standard uniform ancestral and entropy-bounded samplers accept matched
posterior/process or plug-in/process pairs. They use the shared prediction
session so self-conditioning retains one trajectory feedback vector.
