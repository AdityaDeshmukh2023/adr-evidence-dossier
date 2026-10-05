"""Case-level held-out source-label predictions for reproducible paper error analysis."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import CLASSES
from .calibration import softmax
from .inference import _load_bundle, bundle_manifest
from .io import read_json, write_json, object_hash, file_hash


def export_predictions(artifacts_dir, dataset_dir, *, output=None):
    import torch
    artifacts_dir, dataset_dir = Path(artifacts_dir), Path(dataset_dir)
    dataset, splits = read_json(dataset_dir / 'dataset.json'), read_json(dataset_dir / 'splits.json')
    status = bundle_manifest(artifacts_dir)
    if status['status'] != 'available' or status['dataset_sha256'] != dataset['content_sha256']:
        raise ValueError('Error analysis requires a matching model/dataset.')
    manifest, features, model, index, _, _, _, embeddings, calibration, scope = _load_bundle(
        str(artifacts_dir.resolve()), status['manifest_sha256'])
    if scope['split_sha256'] != object_hash(splits):
        raise ValueError('Error analysis split/model mismatch.')
    threshold, rows, conditions = calibration['policy']['threshold'], [], {}
    training_drugs = set(scope['train_drug_ids'])
    with torch.no_grad():
        for condition in ('test', 'test_one_unseen', 'test_both_unseen'):
            positions = splits[manifest['protocol']].get(condition, [])
            if not positions:
                continue
            pairs = torch.tensor([[index[ident] for ident in dataset['pairs'][i]['ingredient_ids']]
                                  for i in positions], dtype=torch.long)
            probabilities = softmax(model.decode(embeddings, pairs).tolist(), calibration['temperature'])
            confusion, undercalls, accepted_errors = Counter(), [], 0
            for position, probability in zip(positions, probabilities):
                pair = dataset['pairs'][position]
                predicted = CLASSES[max(range(3), key=lambda i: probability[i])]
                accepted = threshold is not None and max(probability) >= threshold
                row = {'pair_key': pair['source_record_id'], 'ingredient_ids': pair['ingredient_ids'],
                       'condition': condition, 'source_severity': pair['severity'],
                       'predicted_source_severity': predicted, 'probabilities': dict(zip(CLASSES, probability)),
                       'accepted': accepted, 'confidence': max(probability), 'correct': predicted == pair['severity'],
                       'model_version': manifest['model_version'],
                       'feature_coverage': {'molecules': 2, 'documented_biology_drugs': sum(
                           bool(features['nodes'][index[ident]]['roles'] and features['nodes'][index[ident]]['roles'][0])
                           for ident in pair['ingredient_ids']),
                           'unseen_training_drugs': sum(ident not in training_drugs for ident in pair['ingredient_ids'])}}
                rows.append(row)
                confusion[f"{pair['severity']}->{predicted}"] += 1
                if pair['severity'] == 'high' and predicted != 'high':
                    undercalls.append(pair['source_record_id'])
                accepted_errors += accepted and not row['correct']
            conditions[condition] = {'count': len(positions), 'confusion_counts': dict(confusion),
                                      'high_severity_undercalls': len(undercalls),
                                      'high_undercall_pair_keys': undercalls,
                                      'accepted_errors': accepted_errors}
    result = {'status': 'executed', 'model_version': manifest['model_version'], 'protocol': manifest['protocol'],
              'dataset_sha256': dataset['content_sha256'], 'split_sha256': object_hash(splits),
              'evaluation_code_sha256': file_hash(__file__), 'conditions': conditions, 'predictions': rows,
              'scope': 'Known held-out source labels only; no patient outcomes or unknown-pair discovery claims.'}
    write_json(output or artifacts_dir / 'test_predictions.json', result)
    return result

