"""Likelihood training orchestration for continuous RePlaid.

A GaussianDiffusion recipe with its own loss: adaptive time allocation, the learned
gamma schedule, selected rows' gradient routing and the 9-input RePlaid objective.
Every prediction adds the Gaussian prior's likelihood term to the logits
(``GaussianPriorPosterior.corrected_logits``).
"""

from types import SimpleNamespace

import torch

from core.modifiers.hybrid.views import ViewPoint
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

    def _training_posterior(
        self, state, time, path, selected, bootstrap_path, feedback_transform=None, table=None
    ):
        """The one training prediction path, shared by loss and the endpoint."""
        table = self.representation.codebook() if table is None else table
        return self.prediction.predict(
            self.model,
            self.predictor,
            ((state,), (ViewPoint(time, path),), time),
            state,
            "training_posterior",
            selected=selected,
            mean=self.prediction_mean,
            logits_transform=lambda logits: self._correct(logits, state, path, table, selected),
            bootstrap_feedback_transform=feedback_transform or (lambda value: value),
            bootstrap_logits_transform=lambda logits: self._correct(
                logits, state[selected], bootstrap_path, table
            ),
        )

    def training_prediction(self, states, points, global_time):
        """Training posterior without prompt context: no clean-prompt feedback
        substitution, which loss() applies when ``corrupt_all`` is false."""
        (state,), (local,) = states, points
        path = local.coefficients
        selected = self.prediction.select(len(state), state.device, self.model.training)
        bootstrap_path = SimpleNamespace(alpha=path.alpha[selected], sigma=path.sigma[selected])
        return self._training_posterior(state, local.time, path, selected, bootstrap_path)

    def _prediction(self, pathway, states, points, global_time, trajectory=None):
        (state,), (local,) = states, points
        table = self.representation.codebook()
        return self.prediction.predict(
            self.model,
            self.predictor,
            (states, points, global_time),
            state,
            pathway,
            mean=self.prediction_mean,
            logits_transform=lambda logits: self._correct(logits, state, local.coefficients, table),
            trajectory=trajectory,
        )

    def loss(self, tokens, loss_mask, corruption_mask=None, *, step=0, training=False, **time_context):
        if len(tokens) == 1:
            # Both estimators must be represented even for a singleton batch.
            result = self.loss(
                tokens.expand(2, -1),
                loss_mask.expand(2, -1),
                None if corruption_mask is None else corruption_mask.expand(2, -1),
                step=step,
                training=training,
                **time_context,
            )
            result["loss_sum"] = result["loss_sum"] / 2
            result["num_tokens"] = result["num_tokens"] / 2
            return result
        allocation = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        time, recon = allocation.time, allocation.reconstruction_count
        gamma, derivative, adaptation = self.schedule.gamma(time, recon if training else None)
        clean = self.representation.encode(tokens)
        selected = self.prediction.select(len(tokens), tokens.device, training)
        # Selected (self-conditioned) rows train only the denoiser: their
        # schedule and embedding inputs are detached.
        effective_gamma = detach_selected(gamma, selected)
        effective_derivative = detach_selected(derivative, selected)
        effective_clean = detach_selected(clean, selected)
        path = self.schedule.coefficients(effective_gamma, effective_derivative)
        if self.corrupt_all:
            corruption_mask = torch.ones_like(loss_mask, dtype=torch.bool)
        elif corruption_mask is None:
            corruption_mask = loss_mask
        state = self.process.corrupt(effective_clean, path.alpha, path.sigma, corruption_mask)
        table = self.representation.codebook()
        bootstrap_path = self.schedule.coefficients(
            effective_gamma[selected], effective_derivative[selected]
        )

        def clean_prompt_feedback(feedback):
            # Uncorrupted (prompt) positions feed back their clean state.
            if self.prediction.source != "posterior_mean":
                return feedback
            fixed = ~corruption_mask[selected].bool()[..., None]
            return torch.where(fixed, effective_clean[selected].detach(), feedback)

        log_probs = self._training_posterior(
            state, time, path, selected, bootstrap_path, clean_prompt_feedback, table
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
            self.time_sampler.observe(terms.reconstruction, terms.diffusion, recon, loss_mask)
        weights = loss_mask.to(terms.total)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        total = (terms.total * weights).sum()
        return {"loss": total / count, "loss_sum": total.detach(), "num_tokens": count.detach()}
