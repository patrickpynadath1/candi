import torch
from core.data.sudoku import SudokuTokenizer, encode_example, sudoku_metrics
from core.tasks.sudoku_diversity import remove_hints, canonical_board, diversity_summary
from types import SimpleNamespace


def test_nested_hints_and_answer_preserved():
    grid = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    tokens, mask = encode_example(grid, grid, SudokuTokenizer())
    example = {'input_ids': torch.tensor(tokens), 'attention_mask': torch.tensor(mask)}
    previous = set(range(89))
    for fraction in (1., .75, .5, .25, 0.):
        changed, original, retained = remove_hints(example, fraction, 7)
        puzzle = changed['input_ids'][1:90]
        clues = set(((puzzle >= 1) & (puzzle <= 9)).nonzero().flatten().tolist())
        assert clues <= previous
        assert len(clues) == retained and original == 81
        assert torch.equal(changed['input_ids'][91:], example['input_ids'][91:])
        previous = clues
    assert torch.equal(example['input_ids'], torch.tensor(tokens))


def test_alternative_valid_solutions_and_duplicates():
    tokenizer = SudokuTokenizer()
    grid = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    shifted = [[x % 9 + 1 for x in row] for row in grid]
    tokens, _ = encode_example([[0]*9 for _ in range(9)], grid, tokenizer)
    other, _ = encode_example([[0]*9 for _ in range(9)], shifted, tokenizer)
    target = torch.tensor(tokens[91:]); alternate = torch.tensor(other[91:])
    assert sudoku_metrics(alternate, target, tokens[1:90])['valid_solution']
    rows = [{'generated': tokenizer.decode(x), 'valid_solution': True} for x in (target, target, alternate)]
    rows.append({'generated': '_ '*89, 'valid_solution': False})
    result = diversity_summary(rows, SimpleNamespace(groups=[(0, 20, 0)], repetitions=4))
    assert result['valid_count'] == 3
    assert result['unique_count_within_prompts'] == 2
    assert result['valid_unique_count_within_prompts'] == 2
    assert canonical_board(rows[-1]['generated']) is None
