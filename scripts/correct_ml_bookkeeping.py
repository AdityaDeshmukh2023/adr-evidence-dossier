"""Correct proposed-versus-qualified calibration statistics after all training runs finish."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.artifact_corrections import correct_training_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=ROOT / 'artifacts/ml/training-minibatch')
    parser.add_argument('--deployment', type=Path, default=ROOT / 'artifacts/ml/deployment')
    args = parser.parse_args()
    corrections = correct_training_directory(args.training, args.deployment)
    print(json.dumps({'status': 'executed', 'runs': len(corrections), 'changes': corrections}, indent=2))


if __name__ == '__main__':
    main()
