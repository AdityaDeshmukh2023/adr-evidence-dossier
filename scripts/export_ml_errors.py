"""Export held-out pair predictions and high-severity undercall cases without model tuning."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.error_analysis import export_predictions
from adr_system.ml.io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=ROOT / 'artifacts/ml/training-minibatch')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/ml/dataset')
    args = parser.parse_args()
    summary = read_json(args.training / 'summary.json')
    for run in summary['runs']:
        result = export_predictions(run['path'], args.dataset)
        print(json.dumps({'model': run['model'], 'protocol': run['protocol'], 'seed': run['seed'],
                          'conditions': {key: {k: v for k, v in value.items() if k != 'high_undercall_pair_keys'}
                                         for key, value in result['conditions'].items()}}), flush=True)


if __name__ == '__main__':
    main()
