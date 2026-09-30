"""CANDI with a second, self-conditioned pass on half the training batch."""

import torch

from .candi import CANDI
from .registry import algo_registry


@algo_registry.register("candi_looped")
class CANDILooped(CANDI):
    default_sampler = "candi_looped"
    supported_samplers = frozenset({"candi_looped", "candi_looped_entropy_bounded"})
    sampling_family = "candi_looped"

    def _validate_configuration(self):
        if self.config.algo.backbone != "looped_cont_dit":
            raise ValueError("candi_looped requires backbone=looped_cont_dit")
        loops = self.config.algo.validation_loops
        if isinstance(loops, bool) or not isinstance(loops, int) or loops < 1:
            raise ValueError("algo.validation_loops must be a positive integer")

    def _forward_once(self, noisy_output, previous_hidden=None):
        # CANDI's numerical solver can promote its state to float64. Keep
        # denoiser inputs in the backbone dtype, including preconditioning.
        dtype = self.backbone.vocab_embed.embedding.dtype
        backbone_inputs = {
            key: value.to(dtype=dtype)
            if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in noisy_output.items()
        }
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=torch.cuda.is_available()
        ):
            logits, hidden = self.backbone(
                **backbone_inputs, previous_hidden=previous_hidden, ret_embedding=True
            )
        return self._process_model_output(
            xt=noisy_output["xt"], model_output=logits,
            reveal_mask=noisy_output["reveal_mask"],
        ), hidden

    def forward(self, **kwargs):
        """Standalone denoising for loss evaluation, not the sampling trajectory."""
        hidden = None
        for _ in range(self.config.algo.validation_loops):
            log_probs, hidden = self._forward_once(kwargs, hidden)
            hidden = hidden.detach()
        return log_probs

    def _denoise_for_loss(self, noisy_output, train_mode):
        if not train_mode:
            return self.forward(**noisy_output)
        first, hidden = self._forward_once(noisy_output)
        batch = first.shape[0]
        # Exact half for even batches; stochastic rounding keeps odd batches
        # (including B=1) at 50% in expectation.
        count = batch // 2
        if batch % 2:
            count += int(torch.rand((), device=first.device) < 0.5)
        indices = torch.randperm(batch, device=first.device)[:count]
        if count == 0:
            return first
        subset = {
            key: value.index_select(0, indices)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch
            else value
            for key, value in noisy_output.items()
        }
        second, _ = self._forward_once(
            subset, previous_hidden=hidden.detach().index_select(0, indices)
        )
        # Only the final prediction of each example contributes to its loss.
        return first.index_copy(0, indices, second)
