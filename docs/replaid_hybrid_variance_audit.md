# Hybrid run variance and evaluation failure

The first run (`replaid-hybrid-hard-50k-20260929-161759`) failed at its first
Sudoku evaluation, step 1000. Its final logged training step was 999.

Two integration errors were missed: the new sampler was absent from the
explicit imports in `samplers/registry.py`, and the generic generation API
supplied a nonzero epsilon even though the hybrid sampler requires zero.
Tests imported the sampler directly and explicitly supplied zero, hiding
both failures. Fresh-process registry and public-generation tests now cover
these paths, along with an actual Sudoku callback check.

The original ELBO estimator was unbiased but discarded all predictions
except the randomly selected forced-masked token. The corrected estimator
averages over every masked position. Given the observed mask, the selected
target is uniformly distributed among its masked positions, so this is an
exact Rao–Blackwellization of the previous estimator. It reduces variance
without changing the objective or requiring another model forward pass.
Time draws are now stratified over the learned clock CDF rather than IID.
The learnable clock still receives the per-example second-moment gradient;
this is not an exact optimizer of the stratified whole-batch variance.

The failed run is preserved. No checkpoint was saved before its crash
(the original interval was 5000 steps), so replacement training starts
from scratch. The full derivation describes the corrected estimator.
