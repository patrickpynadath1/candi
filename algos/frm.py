"""Flow Reasoning Models with self-conditioning and Fixed-Point Forcing."""

import torch
import torch.nn.functional as F

from .registry import algo_registry
from .trainer_base import Loss, TrainerBase


GRID_SIDE = 9
GRID_LENGTH = GRID_SIDE * GRID_SIDE
GRID_VOCAB_SIZE = 10
SERIALIZED_LENGTH = 180
PUZZLE_START = 1
SOLUTION_START = 91


def _grid_indices(start, device=None):
    return torch.tensor(
        [start + row * 10 + column for row in range(9) for column in range(9)],
        device=device,
        dtype=torch.long,
    )


def _separator_indices(start, device=None):
    return torch.tensor(
        [start + row * 10 + 9 for row in range(8)],
        device=device,
        dtype=torch.long,
    )


@algo_registry.register("frm")
class FRM(TrainerBase):
    """An 81-cell FRM behind the repository's 180-token Sudoku transport."""

    default_sampler = "frm_euler"
    supported_samplers = frozenset({"frm_euler", "frm_held"})
    sampling_family = "frm"
    sampling_capabilities = frozenset({"frm_denoise", "frm_sudoku_layout"})

    def __init__(self, config, tokenizer):
        # The outer Sudoku tokenizer has BOS/separator symbols. They never enter
        # the paper-matched 81-cell, ten-symbol network.
        super().__init__(config, tokenizer, vocab_size=GRID_VOCAB_SIZE)
        self.grid_length = GRID_LENGTH
        self.training_stage = str(config.algo.training_stage)
        self.self_condition_probability = float(
            config.algo.self_condition_probability
        )
        self.fpf_rollout_probability = float(config.algo.fpf_rollout_probability)
        self.fpf_rollout_depth = int(config.algo.fpf_rollout_depth)
        self.supervised_states = int(config.algo.supervised_states)
        self.clamp_clues = bool(config.algo.clamp_clues)
        self._validate_configuration()

    def _validate_configuration(self):
        if self.config.algo.backbone != "frm_dit":
            raise ValueError("FRM requires the frm_dit backbone")
        if self.config.model.length != SERIALIZED_LENGTH:
            raise ValueError("FRM expects the 180-token Sudoku transport layout")
        if self.training_stage not in {"self_condition", "fpf"}:
            raise ValueError("training_stage must be self_condition or fpf")
        for name, value in (
            ("self_condition_probability", self.self_condition_probability),
            ("fpf_rollout_probability", self.fpf_rollout_probability),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.fpf_rollout_depth < 2 or self.supervised_states < 1:
            raise ValueError("rollout depth and supervised states must be positive")

    def _process_model_input(self, x0, valid_tokens):
        return x0, x0, valid_tokens

    @staticmethod
    def unpack_training_batch(tokens):
        if tokens.ndim != 2 or tokens.shape[1] != SERIALIZED_LENGTH:
            raise ValueError("FRM training input must have shape [batch, 180]")
        puzzle_index = _grid_indices(PUZZLE_START, tokens.device)
        solution_index = _grid_indices(SOLUTION_START, tokens.device)
        clues = tokens.index_select(1, puzzle_index)
        target = tokens.index_select(1, solution_index)
        if torch.any(clues < 0) or torch.any(clues >= GRID_VOCAB_SIZE):
            raise ValueError("puzzle grid contains a non-grid token")
        if torch.any(target < 1) or torch.any(target >= GRID_VOCAB_SIZE):
            raise ValueError("solution grid must contain digits 1..9")
        return clues, target

    @staticmethod
    def clues_from_prompt(prompt_tokens):
        if prompt_tokens.ndim != 2:
            raise ValueError("FRM prompts must have shape [batch, sequence]")
        if prompt_tokens.shape[1] == GRID_LENGTH:
            clues = prompt_tokens
        elif prompt_tokens.shape[1] == SERIALIZED_LENGTH:
            clues = prompt_tokens.index_select(
                1, _grid_indices(PUZZLE_START, prompt_tokens.device)
            )
        else:
            raise ValueError("FRM prompt must contain 81 cells or 180 serialized tokens")
        if torch.any(clues < 0) or torch.any(clues >= GRID_VOCAB_SIZE):
            raise ValueError("FRM clues must use symbols 0..9")
        return clues

    @staticmethod
    def serialize_solution(solution, prompt_tokens=None):
        if solution.ndim != 2 or solution.shape[1] != GRID_LENGTH:
            raise ValueError("FRM solution must have shape [batch, 81]")
        if prompt_tokens is not None and prompt_tokens.shape[1] == GRID_LENGTH:
            return solution
        if prompt_tokens is None:
            output = torch.zeros(
                solution.shape[0], SERIALIZED_LENGTH,
                dtype=solution.dtype, device=solution.device,
            )
        else:
            if prompt_tokens.shape != (solution.shape[0], SERIALIZED_LENGTH):
                raise ValueError("serialized FRM prompt has the wrong shape")
            output = prompt_tokens.clone()
        output[:, SOLUTION_START - 1] = 11  # solution BOS
        output[:, _grid_indices(SOLUTION_START, output.device)] = solution
        output[:, _separator_indices(SOLUTION_START, output.device)] = 10
        return output

    def forward(self, state, time, clues, carry=None):
        time = time.reshape(1) if time.ndim == 0 else time.reshape(-1)
        return self.backbone(state, time, clues, clues.ne(0), carry)

    def predict(self, state, time, clues, carry=None):
        prediction = self.forward(state, time, clues, carry).softmax(-1)
        return self.clamp_prediction(prediction, clues)

    def clamp_state(self, state, clues):
        if not self.clamp_clues:
            return state
        fixed = F.one_hot(clues, GRID_VOCAB_SIZE).to(state)
        return torch.where(clues.ne(0).unsqueeze(-1), fixed, state)

    def clamp_prediction(self, prediction, clues):
        return self.clamp_state(prediction, clues)

    def fpf_rollout(self, noise, target, clues, start_time, end_time):
        """Construct a detached carry using exactly rollout_depth evaluations."""
        target_onehot = F.one_hot(target, GRID_VOCAB_SIZE).float()
        start = start_time[:, None, None]
        state = (1 - start) * noise + start * target_onehot
        state = self.clamp_state(state, clues)
        carry = torch.zeros_like(state)
        denominator_steps = max(self.fpf_rollout_depth - 1, 1)
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                for index in range(self.fpf_rollout_depth):
                    fraction = index / denominator_steps
                    time = start_time + fraction * (end_time - start_time)
                    prediction = self.predict(state, time, clues, carry)
                    if index + 1 < self.fpf_rollout_depth:
                        next_fraction = (index + 1) / denominator_steps
                        next_time = start_time + next_fraction * (end_time - start_time)
                        delta = (next_time - time)[:, None, None]
                        denom = (1 - time)[:, None, None].clamp_min(1e-5)
                        state = state + delta * (prediction - state) / denom
                        state = self.clamp_state(state, clues)
                    carry = prediction
        finally:
            self.backbone.train(was_training)
        return carry.detach()

    def _conventional_carry(self, state, time, clues, train_mode):
        if train_mode and (
            torch.rand((), device=state.device) >= self.self_condition_probability
        ):
            return torch.zeros_like(state)
        with torch.no_grad():
            return self.predict(state, time, clues).detach()

    def _state_loss(self, clues, target, train_mode):
        batch = target.shape[0]
        time = torch.rand(batch, device=target.device)
        noise = torch.randn(
            batch, GRID_LENGTH, GRID_VOCAB_SIZE, device=target.device
        )
        target_onehot = F.one_hot(target, GRID_VOCAB_SIZE).float()
        expanded_time = time[:, None, None]
        supervised_state = (1 - expanded_time) * noise + expanded_time * target_onehot

        use_fpf = (
            train_mode
            and self.training_stage == "fpf"
            and torch.rand((), device=target.device) < self.fpf_rollout_probability
        )
        if use_fpf:
            start_time = torch.rand_like(time) * time
            carry = self.fpf_rollout(noise, target, clues, start_time, time)
        else:
            carry = self._conventional_carry(
                supervised_state, time, clues, train_mode
            )
        logits = self.forward(supervised_state, time, clues, carry)
        return F.cross_entropy(
            logits.reshape(-1, GRID_VOCAB_SIZE), target.reshape(-1), reduction="none"
        ).reshape(batch, GRID_LENGTH), time

    def nll(
        self,
        input_tokens,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        del output_tokens, current_accumulation_step, valid_tokens
        clues, target = self.unpack_training_batch(input_tokens)
        losses = []
        for _ in range(self.supervised_states):
            token_loss, time = self._state_loss(clues, target, train_mode)
            self._record_time_loss(time, token_loss, None, train_mode)
            losses.append(token_loss)
        cell_loss = torch.stack(losses).mean(0)
        serialized = cell_loss.new_zeros(input_tokens.shape)
        serialized[:, _grid_indices(SOLUTION_START, input_tokens.device)] = cell_loss
        return serialized

    def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        del valid_tokens
        token_loss = self.nll(
            x0, x0, current_accumulation_step, train_mode=train_mode
        )
        cell_loss = token_loss.index_select(
            1, _grid_indices(SOLUTION_START, token_loss.device)
        )
        nlls = cell_loss.sum()
        num_tokens = cell_loss.new_tensor(cell_loss.numel())
        return Loss(
            loss=nlls / num_tokens,
            nlls=nlls,
            prior_loss=cell_loss.new_zeros(()),
            num_tokens=num_tokens,
        )

    def prior_sample(self, num_samples, length=None):
        del length
        return torch.randn(
            num_samples, GRID_LENGTH, GRID_VOCAB_SIZE, device=self.device
        )
