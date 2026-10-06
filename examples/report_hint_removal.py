"""Refresh the hint-removal report from completed evaluations (no sampling)."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

ROOT = Path(__file__).resolve().parents[1]


def report(output_root, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Hint-removal sweep: current results', '',
        datetime.now(ZoneInfo('America/Indianapolis')).strftime('Snapshot: %Y-%m-%d %H:%M %Z'), '',
        '**All hints:** success percentage / mean NFE (model calls per sample).', '',
        '**Removed hints:** one number, **valid + unique %** = distinct valid boards / generated samples × 100. Each valid board counts once per evaluation prompt; repetitions add nothing. Invalid boards never count.', '',
        '64 puzzles × 32 generations each, with 5,000-step EMA checkpoints. The GPU inference batch size is 64; the diversity group size is 32 generations per prompt.', '',
        '| Hints retained | Clues retained | Diversity group | Maximum distinct boards counted |',
        '|---|---:|---|---|',
        '| All | 30 | Success rate only | Not a diversity metric |',
        '| 75% | 23 | 32 samples per prompt | Up to 32/32 (100%) per prompt |',
        '| 50% | 15 | 32 samples per prompt | Up to 32/32 (100%) per prompt |',
        '| 25% | 8 | 32 samples per prompt | Up to 32/32 (100%) per prompt |',
        '| None | 0 | All 2,048 samples share the same empty prompt | Up to 2,048/2,048 (100%), pooled |', '',
        'These are sampling-budget ceilings, not enumerated Sudoku solution counts. With clues remaining, the attainable maximum for a prompt is min(32, the number of valid completions); a prompt with one possible solution can score at most 1/32 = 3.125%. Reduced-hint percentages average equally across the 64 groups. At zero hints, duplicates are removed across all 2,048 samples, so one repeated valid board scores 0.049%.', '',
        '`hidden` = normal hidden self-conditioning; `token` = new intermediate token feedback; `none` = no self-conditioning. Regular 32/180 denotes transition steps, not exact NFE. Continuous-only EB is experimental hard commitment. Pending cells have no completed result.', '',
    ]
    rows = []
    for folder in ('hint-removal-discrete-20261005', 'hint-removal-20261005',
                   'sudoku-masked-detach-depth-48k-5k-20261005-hints'):
        if not (output_root / folder / 'manifest.json').exists():
            continue
        manifest = json.loads((output_root / folder / 'manifest.json').read_text())
        for name in sorted({t['run'] for t in manifest['tasks']}):
            lines += ['## ' + name, '',
                '| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |',
                '|---|---:|---:|---:|---:|---:|']
            for task in sorted((t for t in manifest['tasks'] if t['run'] == name), key=lambda t: t['sampler']):
                cells = []
                for fraction in manifest['fractions']:
                    path = Path(task['directory']) / f'hints_{fraction:g}.json'
                    if not path.exists():
                        cells.append('Pending')
                        continue
                    result = json.loads(path.read_text())
                    success = 100 * result['valid_rate']
                    unique = 100 * result['pooled_valid_unique_boards' if fraction == 0 else 'valid_unique_count_within_prompts'] / result['num_generations']
                    if fraction == 1:
                        cells.append(f"{success:.1f}% / {result['mean_nfe']:.2f}")
                    else:
                        cells.append(f'{unique:.3f}%' if 0 < unique < .1 else f'{unique:.1f}%')
                    rows.append(dict(model=name, sampler=task['sampler'], hints_retained=fraction,
                        success_percent=success if fraction == 1 else '',
                        mean_nfe=result['mean_nfe'] if fraction == 1 else '',
                        valid_unique_percent=unique if fraction != 1 else '',
                        diversity_group_size=(result['num_generations'] if fraction == 0 else manifest['repetitions']) if fraction != 1 else '',
                        total_generations=result['num_generations']))
                lines.append('| ' + ' | '.join([task['sampler']] + cells) + ' |')
            lines.append('')
    (report_dir / 'report.md').write_text('\n'.join(lines))
    if rows:
        with (report_dir / 'summary.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f'Refreshed report: {len(rows)} completed cells')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'core/outputs')
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/sudoku-hard-5k/hint-removal")
    args = parser.parse_args()
    report(args.output_root, args.report_dir)
