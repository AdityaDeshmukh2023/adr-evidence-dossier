"""Post-training missing-modality evaluation; never changes model/calibration selection."""
from __future__ import annotations

import math
import random
from pathlib import Path

from . import CLASSES
from .calibration import softmax
from .dataset import audit_splits, audit_predictive_features
from .inference import bundle_manifest, _load_bundle
from .io import read_json, write_json, object_hash, file_hash


def evaluate_missing_modalities(artifacts_dir, dataset_dir, *, fraction=.3, seed=17):
    import torch
    from .metrics import classification_metrics
    dataset_dir, artifacts_dir = Path(dataset_dir), Path(artifacts_dir)
    dataset, splits = read_json(dataset_dir / 'dataset.json'), read_json(dataset_dir / 'splits.json')
    audit_splits(dataset['pairs'], splits)
    audit_predictive_features(dataset)
    status = bundle_manifest(artifacts_dir)
    if status['status'] != 'available' or status['dataset_sha256'] != dataset['content_sha256']:
        raise ValueError('Missing-modality evaluation requires a matching immutable model/dataset.')
    bundle = _load_bundle(str(artifacts_dir.resolve()), status['manifest_sha256'])
    manifest, _, model, index, x, roles, edges, _, calibration, scope = bundle
    if scope['split_sha256'] != object_hash(splits):
        raise ValueError('Missing-modality split/model mismatch.')
    if not 0 < fraction < 1:
        raise ValueError('Use a missing-feature fraction between zero and one.')
    affected = set(random.Random(seed).sample(range(len(index)), math.ceil(len(index) * fraction)))
    molecular_missing = x.clone()
    molecular_missing[list(affected)] = 0
    no_edges = torch.zeros((2, 0), dtype=torch.long)
    variations = {'molecular_zero_30pct_nodes': (molecular_missing, edges, roles, affected)}
    if manifest['model_name'] != 'fingerprint':
        graph_nodes = set(edges.flatten().tolist())
        variations['without_graph_neighbors'] = (x, no_edges, roles, graph_nodes)
    if manifest['model_name'] == 'role_fusion':
        documented = set((roles[:, 0] > 0).nonzero().flatten().tolist())
        variations['without_documented_biology'] = (x, edges, torch.zeros_like(roles), documented)
    conditions = {}
    pairs = torch.tensor([[index[i] for i in row['ingredient_ids']] for row in dataset['pairs']], dtype=torch.long)
    groups = splits[manifest['protocol']]
    with torch.no_grad():
        for condition, (features, graph, biological, missing_nodes) in variations.items():
            embeddings = model.encode(features, graph, biological)
            conditions[condition] = {'changed_node_count': len(missing_nodes), 'total_nodes': len(index),
                                      'changed_node_ids_sha256': object_hash(sorted(missing_nodes)), 'test_conditions': {}}
            for role in ('test', 'test_one_unseen', 'test_both_unseen'):
                if role not in groups or not groups[role]:
                    continue
                positions = groups[role]
                logits = model.decode(embeddings, pairs[positions]).tolist()
                probabilities = softmax(logits, calibration['temperature'])
                labels = [CLASSES.index(dataset['pairs'][i]['severity']) for i in positions]
                affected_pairs = [i for i, position in enumerate(positions)
                                  if any(int(node) in missing_nodes for node in pairs[position])]
                conditions[condition]['test_conditions'][role] = {
                    'total_test_pairs': len(positions), 'affected_test_pairs': len(affected_pairs),
                    'all_pairs': classification_metrics(probabilities, labels, threshold=calibration['policy']['threshold']),
                    'affected_pairs': classification_metrics([probabilities[i] for i in affected_pairs],
                                                            [labels[i] for i in affected_pairs],
                                                            threshold=calibration['policy']['threshold'])}
    return {'status': 'executed', 'model_version': manifest['model_version'], 'model': manifest['model_name'],
            'protocol': manifest['protocol'], 'seed': seed, 'missing_molecule_fraction': fraction,
            'dataset_sha256': dataset['content_sha256'], 'split_sha256': object_hash(splits),
            'evaluation_code_sha256': file_hash(__file__), 'conditions': conditions,
            'scope': 'Controlled missing input features on known source labels; shifted-feature calibration unvalidated. '
                     'Application inference abstains for actual unavailable molecular mappings.'}


def evaluate_training_directory(training_dir, dataset_dir, output):
    training_dir = Path(training_dir)
    summary = read_json(training_dir / 'summary.json')
    runs = []
    for run in summary['runs']:
        result = evaluate_missing_modalities(run['path'], dataset_dir)
        write_json(Path(run['path']) / 'missing_modalities.json', result)
        runs.append(result)
        write_json(output, {'status': 'in_progress', 'runs': runs})
    result = {'status': 'executed', 'runs': runs, 'selected_training': str(training_dir),
              'model_selection_uses_missing_modality_results': False}
    write_json(output, result)
    return result

