"""Train all four ablations on pair and cold-drug protocols; optionally export CPU deployment."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml import MODEL_NAMES
from adr_system.ml.training import train_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/ml/dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ml/training')
    parser.add_argument('--deployment', type=Path, default=ROOT / 'artifacts/ml/deployment')
    parser.add_argument('--protocols', nargs='+', choices=['pair', 'cold'], default=['pair', 'cold'])
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2048, help='Pairs per optimizer step; zero reproduces the full-batch diagnostic.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    result = train_all(args.dataset, args.output, protocols=args.protocols, models=args.models,
                       seeds=args.seeds, epochs=args.epochs, patience=args.patience,
                       device=args.device, deployment=args.deployment,
                       batch_size=args.batch_size,
                       notify=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps({'status': result['status'], 'runs': len(result['runs']),
                      'selected_path': result['selected_path'], 'aggregate': result['aggregate']}, indent=2))


if __name__ == '__main__':
    main()
