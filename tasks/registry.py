"""Routing for datasets that need structured task semantics."""

TASK_DATASETS = frozenset({"sudoku", "tiny_gsm", "gsm8k_test"})


def is_task_dataset(name):
    return name in TASK_DATASETS


def get_task_dataset(config, tokenizer, split):
    name = config.data.train if split == "train" else config.data.valid
    if name == "sudoku":
        from .sudoku import get_sudoku_dataset

        return get_sudoku_dataset(config, tokenizer)[split]
    if name == "tiny_gsm":
        from .tinygsm import get_tinygsm_dataset

        return get_tinygsm_dataset(config, tokenizer)[split]
    if name == "gsm8k_test":
        from .gsm8k import get_gsm8k_dataset

        return get_gsm8k_dataset(config, tokenizer)
    raise ValueError(f"unknown task dataset: {name}")
