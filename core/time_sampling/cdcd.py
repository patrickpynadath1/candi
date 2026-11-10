"""CDCD's learned loss CDF (Appendix A.1--A.4) as a training time sampler.

https://arxiv.org/abs/2211.15089
The fitter is ``core.noise_schedules.learned_time_maps.CDCDTimeMap``; this sampler
draws training times through it: uniform draws on ``[epsilon, 1)`` (antithetic by
default) mapped by the fitted inverse CDF. It is used in training only. A recipe
whose generation also reads the fitted map (view-local times through
``algorithm.adaptive_time_views``) selects the map itself as
``information_path/time_map=cdcd_loss_cdf`` instead, with a uniform sampler.

Legacy loss_derivative checkpoints must not be loaded as this sampler without
explicitly resetting the sampler state. The default
``objective_weighting=uncorrected`` keeps CDCD's time warping as a loss
reweighting: oversampled times also weigh more in the objective.
"""

from core.noise_schedules.learned_time_maps import CDCDTimeMap
from .uniform import UniformTimeSampler
from .density import check_objective_weighting


class CDCDTimeSampler(CDCDTimeMap):
    def __init__(self, adaptive=False, bins=100, min_bin_size=1e-4,
                 learning_rate=1e-3, momentum=0.99, warmup=1000,
                 refit_every=1, epsilon=1e-5, antithetic=True,
                 objective_weighting="uncorrected"):
        super().__init__(adaptive=adaptive, bins=bins, min_bin_size=min_bin_size,
                         learning_rate=learning_rate, momentum=momentum,
                         warmup=warmup, refit_every=refit_every, epsilon=epsilon)
        self.objective_weighting = check_objective_weighting(objective_weighting)
        self.uniform = UniformTimeSampler(epsilon, antithetic)

    def sample(self, count, device, **context):
        return self(self.uniform.sample(count, device, **context))
