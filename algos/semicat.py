from .diffusion import Diffusion
from .registry import algo_registry
import torch
import torch.nn.functional as F
from torch import Tensor


def _view_for(t: Tensor, other: Tensor) -> Tensor:
    """Reshape t to be broadcastable with other (expand trailing dims)."""
    return t.view(other.size(0), *((1,) * (other.ndim - 1)))


@algo_registry.register("semicat")
class Semicat(Diffusion):
    default_sampler = "semicat_flow_map"
    supported_samplers = frozenset({
        "semicat_flow_map", "semicat_euler", "semicat_dopri5",
    })
    sampling_family = "semicat"
    sampling_capabilities = frozenset({
        "continuous_prior", "semicat_flow_map", "semicat_vector_field",
    })
    """Semi-categorical flow matching.

    Gaussian prior -> linear interpolation -> cross-entropy on x1.
    No discrete noise schedule; uses continuous time t in [0, 1].
    """

    def __init__(self, config, tokenizer):
        vocab_size = len(tokenizer)
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

        self.label_smoothing = config.algo.get("label_smoothing", 0.0)
        self.temp = config.algo.get("temp", 1.0)

    def _validate_configuration(self):
        return

    def _backbone_logits(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Run the shared ContDIT while retaining simple test backbones."""
        if hasattr(self.backbone, "reserve_mask_embedding"):
            return self.backbone(
                x,
                sigma,
                reveal_mask=None,
                precond=False,
            )
        return self.backbone(x, sigma)

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def prior_sample(self, *batch_dims) -> Tensor:
        """Sample from isotropic Gaussian prior in R^vocab_size."""
        return torch.randn(
            *batch_dims, self.vocab_size,
            device=self.device, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Linear interpolation: xt = (1-t)*x0 + t*onehot(x1).

        Uses scatter_add for efficiency (equivalent to the one-hot form).

        Args:
            x0: Gaussian sample, shape (B, L, V).
            x1: Clean token indices, shape (B, L).
            t: Time, shape (B, 1, 1) broadcastable with x0.
        Returns:
            xt of shape (B, L, V).
        """
        assert x0.ndim == x1.ndim + 1
        assert t.ndim == x0.ndim
        xt = x0 * (1.0 - t)
        xt.scatter_add_(-1, x1[..., None], t.expand_as(xt[..., :1]))
        return xt

    # ------------------------------------------------------------------
    # Training loss (VFM cross-entropy)
    # ------------------------------------------------------------------
    def nll(self, x0_tokens, output_tokens, current_accumulation_step=None,
            train_mode=False, valid_tokens=None):
        """Compute the flow-matching cross-entropy loss.

        This replaces the standard nll interface. We ignore output_tokens
        since semicat doesn't use a next-token formulation.

        Args:
            x0_tokens: Clean token ids, shape (B, L).
            output_tokens: Unused (kept for API compat).
        Returns:
            Per-token loss of shape (B, L).
        """
        del output_tokens, current_accumulation_step, train_mode

        B, L = x0_tokens.shape
        x1 = x0_tokens  # clean data

        # Sample Gaussian prior noise
        x0 = torch.randn(
            B, L, self.vocab_size, device=x1.device, dtype=torch.float32
        )

        # Sample time uniformly
        t = torch.rand(B, device=x1.device)
        t_broad = _view_for(t, x0)  # (B, 1, 1)

        # Interpolate
        xt = self._interpolate(x0, x1, t_broad)
        if valid_tokens is not None:
            clean = F.one_hot(x1, self.vocab_size).float()
            xt = torch.where(valid_tokens.bool().unsqueeze(-1), xt, clean)

        # Predict clean tokens — network receives (xt, t, t) following
        # the original semicat convention where both time args are t
        # for the VFM (non-distillation) case.
        sigma = self._sigma_from_time(t)
        x1_pred = self._backbone_logits(xt, sigma)

        # Cross-entropy loss per token
        # x1_pred: (B, L, V), x1: (B, L)
        loss = F.cross_entropy(
            x1_pred.transpose(-1, 1),  # (B, V, L)
            x1,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # (B, L)
        return loss

    def _sigma_from_time(self, t: Tensor) -> Tensor:
        """Convert time t to a sigma-like conditioning value.

        The original semicat passes t directly as the sigma/time
        conditioning to the network. We wrap it here so subclasses
        or different backbones can override.
        """
        return t

    def _process_sigma(self, sigma):
        """API compat — semicat uses raw time as conditioning."""
        return sigma

    # ------------------------------------------------------------------
    # Vector field (for ODE sampling)
    # ------------------------------------------------------------------
    def _vector_field(self, xt: Tensor, t: Tensor) -> Tensor:
        """Compute the learned vector field v(xt, t).

        v(xt, t) = (softmax(net(xt, t, t)) - xt) / (1 - t)
        """
        sigma = self._sigma_from_time(t)
        x1_logits = self._backbone_logits(xt, sigma)
        dr = (x1_logits / self.temp).softmax(dim=-1) - xt
        scale = 1.0 / (1.0 - _view_for(t, xt) + 1e-8)
        return dr * scale

    # ------------------------------------------------------------------
    # Flow map (direct jump)
    # ------------------------------------------------------------------
    def _flow_map_step(
        self, x: Tensor, s: Tensor, t: Tensor
    ) -> Tensor:
        """Compute x_{s,t} via the flow map.

        x_{s,t} = x + (t-s)/(1-s) * (softmax(net(x, s, t)) - x)
        In the VFM (non-distillation) case the network receives (x, s, t)
        but the original code just uses (x, s, t) with the same backbone.
        For our backbone that takes (xt, sigma), we condition on s.
        """
        dt = (t - s) / (1.0 - s + 1e-8)
        sigma_s = self._sigma_from_time(s)
        diff = (self._backbone_logits(x, sigma_s) / self.temp).softmax(dim=-1) - x
        return x + _view_for(dt, x) * diff

    def sample_semicat_flow_map(self, *args, **kwargs):
        return self._flow_map_step(*args, **kwargs)

    def sample_semicat_vector_field(self, t, x):
        # torchdiffeq uses (t, state); fixed-step sampling uses the same order.
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self._vector_field(x, t)

    # ------------------------------------------------------------------
    # API compatibility stubs
    # ------------------------------------------------------------------
    def _process_model_input(self, x0, valid_tokens):
        """Semicat doesn't mask or shift tokens."""
        return x0, x0, valid_tokens

    def _process_model_output(self, model_output, xt, sigma):
        """Return log-softmax logits (for eval compatibility)."""
        return (model_output / self.temp).log_softmax(dim=-1)
