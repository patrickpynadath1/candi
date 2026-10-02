"""Likelihood training orchestration for continuous RePlaid."""

import torch

from core.modifiers.self_conditioning.gradient_routing import detach_selected, embedding_mean
from .gaussian_diffusion import GaussianDiffusion


class RePlaid(GaussianDiffusion):
    def __init__(self, *args, corrupt_all=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.representation.spec.unit_normalized:
            raise ValueError("RePlaid requires unit-normalized embeddings")
        self.corrupt_all = corrupt_all

    def _correct(self, logits, state, path, table, selected=None):
        return self.predictor.corrected_logits(
            logits, state, path.alpha, path.sigma, table, selected
        )

    def _prediction(self, pathway, state, time, path, trajectory=None):
        table = self.representation.codebook()
        bootstrap = {}
        if pathway == "training_posterior" and trajectory is None:
            selected = self.prediction.select(
                len(state), state.device, self.model.training
            )
            bootstrap = {
                "selected": selected,
                "bootstrap_logits_transform": lambda logits: (
                    self.predictor.corrected_logits(
                        logits,
                        state[selected],
                        path.alpha[selected],
                        path.sigma[selected],
                        table,
                    )
                ),
            }
        return self.prediction.predict(
            self.model,
            self.predictor,
            self.model_inputs(state, time, path),
            state,
            pathway,
            mean=self.prediction_mean,
            logits_transform=lambda logits: self._correct(logits, state, path, table),
            trajectory=trajectory,
            **bootstrap,
        )

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        **time_context,
    ):
        if len(tokens) == 1:
            # Both estimators must be represented even for a singleton batch.
            tokens = tokens.expand(2, -1)
            loss_mask = loss_mask.expand(2, -1)
            if corruption_mask is not None:
                corruption_mask = corruption_mask.expand(2, -1)
            result = self.loss(
                tokens,
                loss_mask,
                corruption_mask,
                step=step,
                training=training,
                **time_context,
            )
            result["loss_sum"] = result["loss_sum"] / 2
            result["num_tokens"] = result["num_tokens"] / 2
            return result
        allocation = self.time_sampler.sample(
            len(tokens), tokens.device, **time_context
        )
        time, recon = allocation.time, allocation.reconstruction_count
        gamma, derivative, adaptation = self.schedule.gamma(
            time, recon if training else None
        )
        clean = self.representation.encode(tokens)
        selected = self.prediction.select(len(tokens), tokens.device, training)
        effective_gamma = detach_selected(gamma, selected)
        effective_derivative = detach_selected(derivative, selected)
        effective_clean = detach_selected(clean, selected)
        path = self.schedule.coefficients(effective_gamma, effective_derivative)
        if self.corrupt_all:
            corruption_mask = torch.ones_like(loss_mask, dtype=torch.bool)
        elif corruption_mask is None:
            corruption_mask = loss_mask
        state = self.process.corrupt(
            effective_clean, path.alpha, path.sigma, corruption_mask
        )
        table = self.representation.codebook()
        bootstrap_path = self.schedule.coefficients(
            effective_gamma[selected], effective_derivative[selected]
        )
        log_probs = self.prediction.predict(
            self.model,
            self.predictor,
            self.model_inputs(state, time, path),
            state,
            "training_posterior",
            selected=selected,
            mean=self.prediction_mean,
            logits_transform=lambda logits: self._correct(
                logits, state, path, table, selected
            ),
            bootstrap_feedback_transform=lambda feedback: (
                torch.where(
                    ~corruption_mask[selected].bool()[..., None],
                    effective_clean[selected].detach(),
                    feedback,
                )
                if self.prediction.source == "posterior_mean"
                else feedback
            ),
            bootstrap_logits_transform=lambda logits: self._correct(
                logits, state[selected], bootstrap_path, table
            ),
        )
        predicted = embedding_mean(log_probs.exp(), table, selected)
        terminal_gamma, _, _ = self.schedule.gamma(torch.ones(1, device=tokens.device))
        terms = self.objective(
            log_probs,
            tokens,
            effective_clean,
            predicted,
            effective_gamma,
            effective_derivative,
            terminal_gamma,
            clean,
            recon,
        )
        if adaptation is not None:
            adaptation.observe(terms.diffusion, recon, selected, loss_mask)
        if training:
            self.time_sampler.observe(
                terms.reconstruction, terms.diffusion, recon, loss_mask
            )
        weights = loss_mask.to(terms.total)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        total = (terms.total * weights).sum()
        return {
            "loss": total / count,
            "loss_sum": total.detach(),
            "num_tokens": count.detach(),
        }
