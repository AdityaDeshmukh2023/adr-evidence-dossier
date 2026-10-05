"""Measure controlled missing molecular, graph, and biology inputs after model selection."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.robustness import evaluate_training_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=ROOT / 'artifacts/ml/training-minibatch')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/ml/dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ml/training-minibatch/missing_modalities.json')
    args = parser.parse_args()
    result = evaluate_training_directory(args.training, args.dataset, args.output)
    print(json.dumps({'status': result['status'], 'runs': len(result['runs'])}, indent=2))


if __name__ == '__main__':
    main()
