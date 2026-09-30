"""Simplex distribution matching distillation (paper Algorithm 4).

Generator and auxiliary losses have disjoint gradient paths, so a single Adam
step with separate parameter groups implements both updates. The frozen teacher
and all auxiliary weights are retained in checkpoints for resumability.
"""

import copy

import hydra.utils
import torch
from torch import nn
import torch.nn.functional as F

from .registry import algo_registry
from .simplex import SimplexDiffusion
from .simplex_utils import (
    corruption_parameters,
    log_dirichlet_sample,
    dmd_generator_loss,
)


class PairTimeEmbedding(nn.Module):
    def __init__(self, teacher_embedding):
        super().__init__()
        self.target = copy.deepcopy(teacher_embedding)
        self.source = copy.deepcopy(teacher_embedding)
        # Auxiliary initially equals the teacher irrespective of source time.
        self.source.mlp[-1].weight.data.zero_()
        self.source.mlp[-1].bias.data.zero_()

    def forward(self, times):
        return self.source(times[:, 0]) + self.target(times[:, 1])


@algo_registry.register("simplex_dmd")
class SimplexDMD(SimplexDiffusion):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if config.algo.input_mode != "expectation":
            raise ValueError("DMD needs differentiable expectation embeddings")
        self.teacher = copy.deepcopy(self.backbone).requires_grad_(False)
        self.auxiliary = copy.deepcopy(self.backbone)
        self.auxiliary.sigma_map = PairTimeEmbedding(self.backbone.sigma_map)

    def on_fit_start(self):
        # Restored checkpoints already contain the teacher and auxiliary.
        if self.trainer.ckpt_path:
            return
        path = self.config.algo.teacher_checkpoint
        if not path:
            raise ValueError("Simplex DMD training requires algo.teacher_checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        backbone = {
            key[len("backbone.") :]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone)
        if "ema" in checkpoint and self.config.algo.teacher_use_ema:
            shadows = checkpoint["ema"]["shadow_params"]
            parameters = list(self.backbone.parameters())
            if len(shadows) != len(parameters):
                raise ValueError("teacher EMA architecture differs from generator")
            with torch.no_grad():
                for parameter, shadow in zip(parameters, shadows):
                    parameter.copy_(shadow)
        self.teacher.load_state_dict(self.backbone.state_dict())
        aux_state = dict(self.backbone.state_dict())
        for key in list(aux_state):
            if key.startswith("sigma_map."):
                aux_state["sigma_map.target." + key[len("sigma_map.") :]] = (
                    aux_state.pop(key)
                )
        missing, unexpected = self.auxiliary.load_state_dict(aux_state, strict=False)
        if unexpected or any(
            not key.startswith("sigma_map.source.") for key in missing
        ):
            raise ValueError("auxiliary architecture differs from teacher")
        if self.ema:
            from models.ema import ExponentialMovingAverage

            self.ema = ExponentialMovingAverage(
                self._get_parameters(),
                decay=self.config.training.ema,
                use_num_updates=self.config.training.get("ema_use_num_updates", True),
            )
        time_state = {
            key[len("paper_time_sampler.") :]: value
            for key, value in state.items()
            if key.startswith("paper_time_sampler.")
        }
        if time_state:
            self.paper_time_sampler.load_state_dict(time_state)

    def configure_optimizers(self):
        c = self.config
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters(), "lr": c.optim.lr},
                {"params": self.auxiliary.parameters(), "lr": c.algo.auxiliary_lr},
            ],
            betas=(c.optim.beta1, c.optim.beta2),
            eps=c.optim.eps,
            weight_decay=c.optim.weight_decay,
        )
        scheduler = hydra.utils.instantiate(c.lr_scheduler, optimizer=optimizer)
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "name": "trainer/lr"}
        ]

    def _predict_with(self, backbone, log_state, times):
        embeddings = log_state.exp().float() @ backbone.vocab_embed.embedding
        return (
            backbone.forward_with_embeddings(embeddings, times).float().log_softmax(-1)
        )

    def nll(
        self,
        x0,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        if not train_mode:
            return super().nll(
                x0, output_tokens, current_accumulation_step, False, valid_tokens
            )
        # Algorithm 4 leaves Q configurable; use independent adaptive draws.
        s = self._sample_t(len(x0), current_accumulation_step)
        t = self._sample_t(len(x0), current_accumulation_step)
        clean = F.one_hot(x0, self.vocab_size).double()
        with torch.no_grad():
            ps = log_dirichlet_sample(
                corruption_parameters(
                    clean,
                    s.double(),
                    self.concentration(s.double()),
                    self.reference.double(),
                )
            )
            if valid_tokens is not None:
                ps = torch.where(valid_tokens.bool()[..., None], ps, clean.log())
        generated = self(ps, s).exp()
        beta = corruption_parameters(
            generated.double(),
            t.double(),
            self.concentration(t.double()),
            self.reference.double(),
        )
        pt = log_dirichlet_sample(beta)
        if valid_tokens is not None:
            pt = torch.where(valid_tokens.bool()[..., None], pt, clean.log())
        times = torch.stack((s, t), -1)
        self.teacher.eval()
        with torch.no_grad():
            teacher = self._predict_with(self.teacher, pt, t).exp()
            auxiliary = self._predict_with(self.auxiliary, pt, times).exp()
        # Avoid 0*(-inf) at the clean, excluded conditioning positions.
        generator_loss = dmd_generator_loss(
            pt.masked_fill(~torch.isfinite(pt), 0),
            auxiliary,
            teacher,
            self.concentration(t) * (1 - t),
        ).float()
        auxiliary_logp = self._predict_with(self.auxiliary, pt.detach(), times)
        auxiliary_loss = -(auxiliary_logp * generated.detach()).sum(-1)
        self.log("dmd/generator_loss", generator_loss.mean(), on_step=True)
        self.log("dmd/auxiliary_loss", auxiliary_loss.mean(), on_step=True)
        return self.config.algo.generator_weight * generator_loss + auxiliary_loss
