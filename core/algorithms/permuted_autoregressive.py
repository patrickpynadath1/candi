"""Next-token training in a sampled spatial order with decoupled RoPE."""

import torch
from .autoregressive import Autoregressive
from .discrete import reduce_token_loss


class PermutedAutoregressive(Autoregressive):
    def __init__(
        self,
        model,
        predictor,
        objective,
        order,
        length=180,
        condition_on_observed=False,
    ):
        super().__init__(model, predictor, objective, length)
        self.order = order
        self.condition_on_observed = condition_on_observed

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
        if tokens.shape[1] < 2:
            raise ValueError(
                "permuted autoregressive training requires at least two tokens"
            )
        order = self.order.sample(
            len(tokens), tokens.shape[1] - 1, device=tokens.device
        )
        if self.condition_on_observed:
            if corruption_mask is None:
                raise ValueError("observed conditioning requires corruption_mask")
            order = self.order.observed_first(order, ~corruption_mask[:, 1:].bool())
        targets = tokens[:, 1:].gather(1, order)
        inputs = torch.cat((tokens[:, :1], targets[:, :-1]), 1)
        key_positions = torch.cat(
            (torch.zeros_like(order[:, :1]), order[:, :-1] + 1), 1
        )
        prediction = self.predictor.training_posterior(
            self.model(
                inputs, key_position_ids=key_positions, query_position_ids=order + 1
            ),
            None,
        )
        return reduce_token_loss(
            self.objective(prediction, targets), loss_mask[:, 1:].gather(1, order)
        )

    def sampling_prediction(
        self, tokens, cache=None, *, key_position_ids, query_position_ids
    ):
        return self.predictor.sampling_posterior(
            self.model(
                tokens,
                cache=cache,
                key_position_ids=key_position_ids,
                query_position_ids=query_position_ids,
            ),
            None,
        )
