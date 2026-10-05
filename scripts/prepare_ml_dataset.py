"""Freeze molecular features and leakage-audited pair/cold-drug research splits."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/processed_ddinter/ddinter_cleaned.csv')
    parser.add_argument('--molecules', type=Path, default=ROOT / 'artifacts/ml/molecules.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ml/dataset')
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.source, args.molecules, args.output, seed=args.seed), indent=2))


if __name__ == '__main__':
    main()
