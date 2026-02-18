from .diffusion import Diffusion


import torch


class UniformState(Diffusion):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.time_conditioning
        assert self.parameterization == "mean"
        if self.config.algo.name != "distillation":
            assert self.T == 0

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape
            (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)
        xt = torch.where(move_indices, uniform_tensor, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        return torch.randint(
            0, self.vocab_size, batch_dims, dtype=torch.int64, device=self.device
        )