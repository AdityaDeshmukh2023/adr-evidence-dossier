"""Exercise actual ML training on a bounded real-source subset; not full validation."""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.dataset import prepare_dataset, make_splits
from adr_system.ml.io import read_json, write_json, object_hash
from adr_system.ml.training import train_run
from adr_system.ml.inference import bundle_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecules', type=Path, default=ROOT / 'data/ml_smoke_molecules.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ml/smoke')
    parser.add_argument('--per-class', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    prepare_dataset(ROOT / 'data/processed_ddinter/ddinter_cleaned.csv', args.molecules,
                    args.output / 'data')
    dataset = read_json(args.output / 'data/dataset.json')
    buckets = defaultdict(list)
    for row in dataset['pairs']:
        buckets[row['severity']].append(row)
    rng = random.Random(1729)
    chosen = []
    for severity in ('high', 'moderate', 'low'):
        if len(buckets[severity]) < 30:
            raise ValueError(f'Need at least thirty mapped real {severity} pairs for a useful smoke.')
        chosen.extend(rng.sample(buckets[severity], min(args.per_class, len(buckets[severity]))))
    dataset['pairs'] = sorted(chosen, key=lambda row: row['source_record_id'])
    dataset['class_counts'] = dict(Counter(row['severity'] for row in dataset['pairs']))
    dataset['scope'] = 'Limited real-source training smoke; not full benchmark or clinical validation.'
    dataset['content_sha256'] = object_hash({k: v for k, v in dataset.items() if k != 'content_sha256'})
    splits = make_splits(dataset['pairs'], chemical_keys={row['ingredient_id']: row['inchikey'] for row in dataset['nodes']})
    splits['dataset_sha256'] = dataset['content_sha256']
    write_json(args.output / 'data/dataset.json', dataset)
    write_json(args.output / 'data/splits.json', splits)
    rows = []
    for model_name in ('fingerprint', 'graphsage', 'fusion', 'role_fusion'):
        result = train_run(dataset, splits, args.output / model_name, model_name=model_name,
                           epochs=args.epochs, seed=17, purpose='limited_real_data_training_smoke',
                           notify=lambda value: print(json.dumps(value), flush=True))
        rows.append({'model': model_name, 'seconds': result['training_seconds'],
                     'validation_macro_f1': result['best_validation_macro_f1'],
                     'bundle': bundle_manifest(args.output / model_name)})
    result = {'status': 'executed', 'scope': dataset['scope'], 'pairs': len(dataset['pairs']),
              'epochs': args.epochs, 'runs': rows}
    write_json(args.output / 'smoke_results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
