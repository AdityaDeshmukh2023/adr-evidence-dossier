"""Run real SHAP or GNNExplainer and measured deletion/stability checks on held-out pairs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.explain import run_explanations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/ml/deployment')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/ml/dataset')
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--method', choices=['auto', 'shap', 'gnn'], default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--stability-cases', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_explanations(args.artifacts, args.dataset, count=args.count, method=args.method,
                              epochs=args.epochs, samples=args.samples,
                              stability_cases=args.stability_cases, output=args.output)
    print(json.dumps({k: value for k, value in result.items() if k != 'cases'}, indent=2))
    return int(result['status'] != 'executed')


if __name__ == '__main__':
    raise SystemExit(main())
