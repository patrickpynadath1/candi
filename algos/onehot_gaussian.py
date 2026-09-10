"""Pure Gaussian interpolation in one-hot space, with a single adaptive clock."""

import torch
import torch.nn.functional as F

from .candi_embedding import CANDIEmbedding
from .rank_schedules import onehot_rank
from .registry import algo_registry


@algo_registry.register('onehot_gaussian')
class OnehotGaussian(CANDIEmbedding):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if self.hybrid_mask or self.interpolation_schedule != 'linear':
            raise ValueError('onehot_gaussian requires pure continuous linear interpolation')

    def clean_state(self, x):
        return F.one_hot(x, self.vocab_size).float() if x.ndim == 2 else x

    def prior_sample(self, *batch_dims):
        return torch.randn(*batch_dims, self.vocab_size, device=self.device)

    @torch.no_grad()
    def rank_diagnostics(self):
        t = torch.linspace(0, 1, 101, device=self.device)
        rho = 1-self.noise.alpha_t(t)
        sigma = rho/(1-rho)
        return dict(clock=t.tolist(), progress=rho.tolist(), interpolation=rho.tolist(),
                    rank=onehot_rank(sigma).tolist(), space='onehot',
                    calibration='analytic Gaussian', prior='standard Gaussian',
                    prior_matches_endpoint=True,
                    rank_condition='continuously corrupted positions')
