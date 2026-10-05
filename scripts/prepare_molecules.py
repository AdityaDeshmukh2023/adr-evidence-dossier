"""Map the complete catalog to PubChem with a resumable public-data cache."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.molecules import prepare_molecules
from adr_system.terminology import catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ml/molecules.json')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--max-consecutive-errors', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    result = prepare_molecules(catalog()[1], args.output, limit=args.limit,
                               retry_failed=args.retry_failed, timeout=args.timeout,
                               max_consecutive_errors=args.max_consecutive_errors,
                               workers=args.workers,
                               notify=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
    return int(result['status'] == 'network_interrupted')


if __name__ == '__main__':
    raise SystemExit(main())
