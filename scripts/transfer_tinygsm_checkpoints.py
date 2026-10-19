#!/usr/bin/env python3
"""Transfer the latest checkpoints from the seven selected TinyGSM runs.

Preview:
  python3 scripts/transfer_tinygsm_checkpoints.py
Copy using ~/move_checkpoints.sh:
  python3 scripts/transfer_tinygsm_checkpoints.py --send

The existing helper places each file under its configured remote base directory:
  <run-name>/job-<slurm-id>-step-<global-step>/checkpoints/<original-filename>
No local checkpoints are moved or deleted.
"""

import argparse
from pathlib import Path
import shlex
import pickletools
import zipfile
import subprocess

RUNS = (
    'tinygsm-block64-embed-linear-no-noise-warmup',
    'tinygsm-block64-embed-linear_rank',
    'tinygsm-block64-embed-linear',
    'tinygsm-hybrid-embed-linear',
    'tinygsm-hybrid-embed-adaptive_linear_rank',
    'tinygsm-hybrid-embed-linear_rank',
    'tinygsm-hybrid-embed-geometric_rank',
)


def checkpoint_step(path):
    # Inspect metadata without unpickling historical project classes or loading
    # multi-GB tensors. Lightning writes global_step in its checkpoint header.
    with zipfile.ZipFile(path) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(metadata) != 1:
            raise ValueError(f'Expected one PyTorch metadata record: {path}')
        operations = pickletools.genops(archive.read(metadata[0]))
        for opcode, value, _ in operations:
            if opcode.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE') and value == 'global_step':
                for opcode, value, _ in operations:
                    if opcode.name in ('BINPUT', 'LONG_BINPUT', 'MEMOIZE', 'PUT'):
                        continue
                    if opcode.name in ('BININT', 'BININT1', 'BININT2', 'INT', 'LONG', 'LONG1', 'LONG4') and isinstance(value, int):
                        return value
                    raise ValueError(f'Unexpected global_step encoding: {path}')
    raise ValueError(f'Missing global_step: {path}')


def latest_checkpoint(directory):
    candidates = []
    for path in sorted(directory.glob('*/checkpoints/*.ckpt')):
        step = checkpoint_step(path)
        candidates.append((step, path.name == 'last.ckpt', path.stat().st_mtime_ns, path))
    if not candidates:
        raise FileNotFoundError(f'No checkpoints found in {directory}')
    step, _, _, path = max(candidates)
    return step, path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path('/scratch/gautschi/ppynadat/cont_diff/candi-v2'))
    parser.add_argument('--helper', type=Path, default=Path.home() / 'move_checkpoints.sh')
    parser.add_argument('--send', action='store_true', help='Actually copy; otherwise only print the plan')
    args = parser.parse_args()
    if not args.helper.is_file():
        parser.error(f'Transfer helper not found: {args.helper}')
    # Resolve every run before sending anything, to catch missing data up front.
    plan = []
    for name in RUNS:
        step, source = latest_checkpoint(args.root / name)
        destination = f'{name}/job-{source.parent.parent.name}-step-{step}'
        # The existing helper uses unquoted shell variables for its arguments.
        if any(c.isspace() or c in "'\"$`\\;|&<>()*?[]{}" for c in str(source) + destination):
            raise ValueError(f'Path unsupported by the existing transfer helper: {source}')
        command = ['bash', '-e', str(args.helper), destination, str(source)]
        plan.append(command)
        print(f'{name}: step {step:,}\n  {shlex.join(command)}', flush=True)
    if not args.send:
        print('\nPreview only. Add --send to copy all seven checkpoints.')
        return
    for command in plan:
        # -e makes failures propagate even though the helper ends with echo.
        subprocess.run(command, check=True)
    print('\nTransferred all seven checkpoints.')


if __name__ == '__main__':
    main()
