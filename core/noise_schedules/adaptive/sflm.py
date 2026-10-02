import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from .clock import BufferedAdaptiveSchedule


class SFLMAdaptiveSchedule(BufferedAdaptiveSchedule):
    """Reference S-FLM spline adaptation; sampled-time coordinates by default."""

    def _refit(self):
        model = make_pipeline(
            SplineTransformer(
                n_knots=self.n_knots,
                degree=self.spline_degree,
                extrapolation="continue",
            ),
            Ridge(alpha=self.ridge_alpha),
        )
        model.fit(
            self.t_buf.cpu().numpy().reshape(-1, 1),
            self.loss_buf.cpu().numpy(),
        )
        smooth_loss = model.predict(self._grid.reshape(-1, 1))
        importance = np.maximum(np.gradient(smooth_loss, self._grid), 0)
        importance = (1 - self.uniform_mix) * importance + self.uniform_mix
        self.loss_profile.copy_(torch.from_numpy(smooth_loss))
        self.sampling_density.copy_(torch.from_numpy(importance / importance.sum()))
        if self.coordinate == "physical":
            # Integrate intervals so F(0)=0 exactly; pointwise cumsum followed
            # by extrapolation can otherwise take the schedule out of bounds.
            cdf = np.concatenate(
                ([0.0], np.cumsum((importance[:-1] + importance[1:]) / 2))
            )
        else:
            cdf = np.cumsum(importance)
        cdf /= cdf[-1]
        remapped_t = PchipInterpolator(cdf, self._grid)(self._grid)
        if self.coordinate == "physical":
            hi = self.base.alpha_t(torch.tensor(0.0, dtype=torch.float64)).item()
            lo = self.base.alpha_t(torch.tensor(1.0, dtype=torch.float64)).item()
            alpha = hi + (lo - hi) * remapped_t
            alpha[0], alpha[-1] = hi, lo
        else:
            alpha = self.base.alpha_t(torch.as_tensor(remapped_t)).cpu().numpy()
        alpha_tensor = torch.from_numpy(alpha).to(self.ema_alpha_vals)
        self.refit_count += 1
        if self.ema > 0:
            self.ema_alpha_vals.mul_(self.ema).add_(alpha_tensor, alpha=1 - self.ema)
            correction = 1 - self.ema ** self.refit_count.item()
            alpha = (self.ema_alpha_vals / correction).cpu().numpy()
        if self.coordinate == "physical":
            alpha[0], alpha[-1] = hi, lo
        self._alpha_spline = PchipInterpolator(self._grid, alpha)
        self._dalpha_spline = self._alpha_spline.derivative()
        self.alpha_vals.copy_(torch.from_numpy(alpha))
        self.has_schedule.fill_(True)
