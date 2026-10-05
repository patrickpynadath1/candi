"""Train the masked + Gaussian smoke check, then evaluate its EMA checkpoint."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps import compose
from core.training.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    config = compose('sudoku_hybrid_masked_gaussian_check', {'output_dir': str(output)})
    if not args.evaluate_only:
        run(config)
    summaries = {}
    for steps in (32, 180):
        evaluation = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        evaluation.mode = 'evaluate'
        evaluation.wandb = None
        evaluation.output_dir = str(output / f'evaluation_ema_{steps}')
        evaluation.evaluation.checkpoint = str(output / 'checkpoints/last.ckpt')
        evaluation.evaluation.use_ema = True
        evaluation.evaluation.max_examples = 2000
        evaluation.sampler.steps = steps
        run(evaluation)
        result = json.loads((Path(evaluation.output_dir) / 'results.json').read_text())
        summaries[str(steps)] = {key: value for key, value in result.items() if key != 'records'}
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('FINAL EVALUATION', json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
