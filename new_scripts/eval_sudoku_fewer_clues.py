#!/usr/bin/env python3
"""Evaluate nested, deterministically thinned hard Sudoku prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys


def thin_tokens(tokens, index, clues):
    tokens = list(tokens)
    positions = [i for i in range(1, 90) if 1 <= tokens[i] <= 9]
    if len(positions) != 30:
        raise ValueError(f'Expected 30 original clues, got {len(positions)}')
    random.Random(420000 + index).shuffle(positions)
    for i in positions[clues:]:
        tokens[i] = 0
    return tokens


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--clues', type=int, choices=range(31), required=True)
    args, overrides = parser.parse_known_args()
    sys.path.insert(0, str(args.source.resolve()))
    sys.argv = [str(args.source / 'main.py'), '--config-path',
                str(args.source.resolve() / 'configs'), *overrides]
    import main as entry
    import tasks.evaluation as evaluation
    from datasets import Dataset, DatasetDict
    original = evaluation.get_sudoku_dataset

    def thinned_dataset(config, tokenizer):
        data = original(config, tokenizer)['validation'].with_format(None)
        digest = hashlib.sha256()
        def transform(row, index):
            tokens = thin_tokens(row['input_ids'], index, args.clues)
            assert tokens[90:] == row['input_ids'][90:]
            digest.update(bytes(tokens[:90]))
            return {'input_ids': tokens}
        valid = data.map(transform, with_indices=True, load_from_cache_file=False,
                         keep_in_memory=True, desc=f'Retaining {args.clues} clues')
        output = Path(config.sudoku.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'clue_audit.json').write_text(json.dumps(dict(
            clues=args.clues, examples=len(valid), original_clues=30,
            selection_seed='420000 + validation index', nested_subsets=True,
            prompt_sha256=digest.hexdigest(), targets_unchanged=True,
            primary_metric='valid_solution_rate: valid grid AND clue consistency AND valid format'
        ), indent=2))
        return DatasetDict(validation=valid).with_format('torch')
    evaluation.get_sudoku_dataset = thinned_dataset
    entry.main()


if __name__ == '__main__':
    main()
