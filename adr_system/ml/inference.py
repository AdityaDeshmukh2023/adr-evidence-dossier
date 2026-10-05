"""CPU-only research inference with artifact checks and source-unknown gating."""
from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path

from . import CLASSES, MODEL_NAMES, TASK
from .calibration import softmax
from .io import file_hash, read_json

REQUIRED_FILES = {'weights.pt', 'features.json', 'graph.json', 'config.json',
                  'calibration.json', 'training_scope.json'}
OPTIONAL_FILES = {'explanations.json'}


def bundle_manifest(artifacts_dir) -> dict:
    directory = Path(artifacts_dir)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        return {'status': 'unavailable', 'reason': 'model_artifacts_absent', 'task': TASK}
    try:
        manifest = read_json(manifest_path)
        if (manifest.get('schema') != 'model-bundle-1' or manifest.get('task') != TASK
                or manifest.get('classes') != list(CLASSES) or manifest.get('model_name') not in MODEL_NAMES
                or not REQUIRED_FILES <= set(manifest.get('files', {}))
                or not set(manifest.get('files', {})) <= REQUIRED_FILES | OPTIONAL_FILES):
            raise ValueError('unsupported_model_bundle_schema')
        hashes = {}
        for name, expected in manifest['files'].items():
            # Exact allowed filenames prevent untrusted manifests reading other paths.
            path = directory / name
            if not path.is_file():
                raise ValueError('required_model_file_absent')
            actual = file_hash(path)
            if actual != expected:
                raise ValueError('model_artifact_hash_mismatch')
            hashes[name] = actual
        from adr_system.data import DATA_DIR
        source_path = DATA_DIR / 'processed_ddinter/ddinter_cleaned.csv'
        source_compatible = source_path.is_file() and file_hash(source_path) == manifest.get('source_csv_sha256')
        from adr_system.biology import snapshot_manifest
        current_biology = snapshot_manifest()
        biology_compatible = current_biology.get('roles_sha256') == manifest.get('biology_manifest', {}).get('roles_sha256')
        return {'status': 'available', 'task': TASK, 'model_version': manifest['model_version'],
                'model_name': manifest['model_name'], 'protocol': manifest['protocol'],
                'training_scope': manifest['training_scope'], 'manifest_sha256': file_hash(manifest_path),
                'hashes': hashes, 'dataset_sha256': manifest.get('dataset_sha256'),
                'source_csv_sha256': manifest.get('source_csv_sha256'),
                'molecule_cache_sha256': manifest.get('molecule_cache_sha256'),
                'biology_manifest': manifest.get('biology_manifest', {}),
                'runtime_available': importlib.util.find_spec('torch') is not None,
                'source_snapshot_compatible': source_compatible,
                'biology_snapshot_compatible': biology_compatible,
                'prediction_semantics': manifest['prediction_semantics']}
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'invalid', 'reason': 'invalid_or_changed_model_artifacts', 'task': TASK}


@lru_cache(maxsize=2)
def _load_bundle(directory: str, manifest_hash: str):
    import torch
    from .models import PairClassifier, tensors
    path = Path(directory)
    manifest = read_json(path / 'manifest.json')
    features = read_json(path / 'features.json')
    config = read_json(path / 'config.json')
    calibration = read_json(path / 'calibration.json')
    scope = read_json(path / 'training_scope.json')
    graph = read_json(path / 'graph.json')
    if (features['fingerprint'] != {'method': 'Morgan', 'radius': 2, 'bits': 2048, 'include_chirality': True}
            or config['model_name'] != manifest['model_name'] or config['fingerprint_dim'] != 2048
            or calibration['protocol'] != manifest['protocol']
            or scope['purpose'] != manifest['training_scope']):
        raise ValueError('Model artifact configuration mismatch.')
    for row in features['nodes']:
        if (any(not isinstance(bit, int) or bit < 0 or bit >= 2048 for bit in row['fingerprint_bits'])
                or len(row['roles']) != config['role_dim']
                or any(not math.isfinite(float(value)) for value in row['roles'])):
            raise ValueError('Invalid frozen molecular/biology features.')
    if len({row['ingredient_id'] for row in features['nodes']}) != len(features['nodes']):
        raise ValueError('Duplicate model ingredient IDs.')
    graph_keys = {'|'.join(sorted(pair)) for pair in graph['pairs']}
    if (len(graph_keys) != len(graph['pairs'])
            or graph_keys != set(scope['message_pair_keys'])):
        raise ValueError('Model graph/training provenance mismatch.')
    temperature = float(calibration['temperature'])
    threshold = calibration['policy']['threshold']
    if (not math.isfinite(temperature) or temperature <= 0
            or (threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1))):
        raise ValueError('Invalid model calibration.')
    model = PairClassifier(**{key: value for key, value in config.items() if key != 'training'})
    torch.set_num_threads(min(4, torch.get_num_threads()))
    state = torch.load(path / 'weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    index, x, roles, _, edges = tensors(features, graph['pairs'])
    with torch.no_grad():
        embeddings = model.encode(x, edges, roles)
    return manifest, features, model, index, x, roles, edges, embeddings, calibration, scope


def predict_pairs(ingredient_pairs, artifacts_dir) -> list[dict]:
    """Conditional-severity research API; callers must verify documented source status."""
    status = bundle_manifest(artifacts_dir)
    base = {'task': TASK, 'conditional_on': 'documented interaction',
            'status': 'abstained', 'model_version': status.get('model_version'),
            'model_name': status.get('model_name'), 'calibration_protocol': status.get('protocol'),
            'scope': 'Predict source severity; not patient risk or proof of a new interaction.'}
    output = [{**base, 'ingredient_ids': sorted(pair), 'source_record_id': '|'.join(sorted(pair)),
               'source_state': 'not_checked_by_research_api', 'accepted': False, 'predicted_severity': None,
               'probabilities': {}, 'explanation': None, 'explanation_reference': None,
               'explanation_status': 'not_precomputed',
               'calibration_sha256': status.get('hashes', {}).get('calibration.json'),
               'calibration_version': status.get('hashes', {}).get('calibration.json'),
               'feature_coverage': {'molecular': 'unavailable', 'biology': 'unknown'},
               'abstention_reason': None}
              for pair in ingredient_pairs]
    if status['status'] != 'available':
        for row in output:
            row['abstention_reason'] = status['reason']
        return output
    if not status['runtime_available']:
        for row in output:
            row['abstention_reason'] = 'optional_torch_runtime_unavailable'
        return output
    if not status['source_snapshot_compatible'] or not status['biology_snapshot_compatible']:
        for row in output:
            row['abstention_reason'] = 'source_or_biology_snapshot_changed_since_training'
        return output
    try:
        import torch
        bundle = _load_bundle(str(Path(artifacts_dir).resolve()), status['manifest_sha256'])
        manifest, features, model, index, _, _, _, embeddings, calibration, scope = bundle
        for row in output:
            ids = row['ingredient_ids']
            mapped = sum(ident in index for ident in ids)
            row['feature_coverage'] = {'molecular': 'complete' if mapped == 2 else 'partial' if mapped else 'unavailable',
                                       'mapped_drugs': mapped, 'requested_drugs': 2,
                                       'biology': 'documented_for_both' if mapped == 2 and all(
                                           features['nodes'][index[ident]]['roles'] and features['nodes'][index[ident]]['roles'][0]
                                           for ident in ids) else 'bounded_snapshot_partial_or_unknown'}
            if len(ids) != 2 or ids[0] == ids[1] or any(ident not in index for ident in ids):
                row['abstention_reason'] = 'molecular_mapping_unavailable'
                continue
            if scope['purpose'] != 'source_label_benchmark':
                row['abstention_reason'] = 'smoke_artifact_not_validated_for_application'
                continue
            if manifest['protocol'] != 'pair' or any(ident not in scope['train_drug_ids'] for ident in ids):
                row['abstention_reason'] = 'calibration_protocol_does_not_cover_unseen_drug'
                continue
            query = torch.tensor([[index[ident] for ident in ids]], dtype=torch.long)
            with torch.no_grad():
                logits = model.decode(embeddings, query).tolist()
            probabilities = softmax(logits, calibration['temperature'])[0]
            prediction = max(range(3), key=lambda i: probabilities[i])
            threshold = calibration['policy']['threshold']
            row['probabilities'] = dict(zip(CLASSES, probabilities))
            row['predicted_severity'] = CLASSES[prediction]
            row['accepted'] = threshold is not None and max(probabilities) >= threshold
            row['status'] = 'accepted' if row['accepted'] else 'abstained'
            row['abstention_reason'] = (None if row['accepted'] else
                                        'no_calibration_threshold_qualifies' if threshold is None else
                                        'prediction_below_calibrated_threshold')
            explanation_path = Path(artifacts_dir) / 'explanations.json'
            if row['accepted'] and 'explanations.json' in status['hashes']:
                explanations = read_json(explanation_path)
                entry = next((entry for entry in explanations.get('cases', [])
                              if entry.get('pair_key') == row['source_record_id']), None)
                if entry:
                    row['explanation'] = entry
                    row['explanation_status'] = 'available' if entry.get('status') == 'executed' else 'failed'
                    row['explanation_reference'] = {'file': 'explanations.json',
                                                     'sha256': status['hashes']['explanations.json'],
                                                     'pair_key': row['source_record_id']}
    except ImportError:
        for row in output:
            row.update(status='abstained', accepted=False, predicted_severity=None, probabilities={},
                       explanation=None, explanation_reference=None, explanation_status='failed',
                       abstention_reason='optional_ml_dependency_unavailable')
    except Exception:
        # An invalid optional model must never interrupt authoritative source screening.
        for row in output:
            row.update(status='abstained', accepted=False, predicted_severity=None, probabilities={},
                       explanation=None, explanation_reference=None, explanation_status='failed',
                       abstention_reason='invalid_or_incompatible_model_artifacts')
    return output


def predict_unknown(result, artifacts_dir) -> list[dict]:
    """Only documented pairs whose current source severity is unknown are eligible."""
    from adr_system.knowledge import pair_state
    assessments = result.get('pair_assessments', []) if isinstance(result, dict) else result.pair_assessments
    pairs, seen = [], set()
    for assessment in assessments:
        ids = assessment.get('ingredient_ids', [])
        if (assessment.get('state') != 'unknown' or not isinstance(ids, (list, tuple)) or len(ids) != 2
                or not all(isinstance(ident, str) for ident in ids)):
            continue
        key = tuple(sorted(ids))
        if key in seen:
            continue
        try:
            if pair_state(*key) != 'unknown':
                continue
        except (TypeError, ValueError):
            continue
        seen.add(key)
        pairs.append(list(key))
    rows = predict_pairs(pairs, artifacts_dir) if pairs else []
    for row in rows:
        row['source_state'] = 'unknown'
        row['source_unknown_applicability'] = 'unvalidated_distribution; source labels unavailable'
    return rows
