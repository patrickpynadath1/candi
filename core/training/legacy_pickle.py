"""Read trusted legacy checkpoints without depending on obsolete tokenizers."""

import pickle

from core.data.sudoku import SudokuTokenizer


class _TokenizerMetadata:
    """Inert saved hyperparameter, never used to tokenize in core.

    Transformers 5 removed the old GPT2TokenizerFast module path. Checkpoint
    import only needs weights; the configured Task constructs its tokenizer.
    """

    def __setstate__(self, state):
        self.saved_state = state

    def __getstate__(self):
        return self.saved_state


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == (
            "transformers.models.gpt2.tokenization_gpt2_fast",
            "GPT2TokenizerFast",
        ):
            return _TokenizerMetadata
        if (module, name) in {
            ("tasks.sudoku", "SudokuTokenizer"),
            ("dataloader", "SudokuTokenizer"),
        }:
            return SudokuTokenizer
        if module.split(".")[0] in {
            "algos",
            "models",
            "samplers",
            "tasks",
            "dataloader",
            "utils",
            "metrics",
            "main",
        }:
            raise ValueError(f"unsupported legacy checkpoint object: {module}.{name}")
        return super().find_class(module, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()
