"""Reproducible conditional-severity baselines, calibration, and CPU bundles."""
from __future__ import annotations

import copy
import random
import statistics
import time
import sys
import importlib.metadata
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

from . import CLASSES, MODEL_NAMES, TASK
from .calibration import fit_temperature, softmax, choose_threshold, wilson_upper
from .dataset import audit_splits, audit_predictive_features, _partition
from .io import read_json, write_json, object_hash, file_hash


def code_snapshot() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'adr_system/ml').glob('*.py'))
    paths += [root / 'scripts/train_ml.py', root / 'scripts/prepare_ml_dataset.py', root / 'scripts/prepare_molecules.py']
    return {'captured_at': datetime.now(timezone.utc).isoformat(), 'capture_point': 'before_training_function_execution',
            'files': {str(path.relative_to(root)).replace('\\', '/'): file_hash(path) for path in paths}}


def write_bundle(output, model, config, dataset, graph_pairs, calibration, scope, *, seed,
                 protocol, model_version=None):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'weights.pt')
    features = {key: dataset[key] for key in ('nodes', 'fingerprint', 'role_feature_names')}
    write_json(output / 'features.json', features)
    write_json(output / 'graph.json', {'pairs': graph_pairs, 'edge_type': 'untyped_training_only'})
    write_json(output / 'config.json', config)
    write_json(output / 'calibration.json', calibration)
    write_json(output / 'training_scope.json', scope)
    files = {name: file_hash(output / name) for name in ('weights.pt', 'features.json', 'graph.json',
                                                        'config.json', 'calibration.json', 'training_scope.json')}
    manifest = {'schema': 'model-bundle-1', 'task': TASK, 'classes': list(CLASSES),
                'model_name': config['model_name'], 'model_version': model_version or object_hash(files)[:20],
                'seed': seed, 'protocol': protocol, 'files': files,
                'dataset_sha256': dataset.get('content_sha256'),
                'source_csv_sha256': dataset.get('source_csv_sha256'),
                'molecule_cache_sha256': dataset.get('molecule_cache_sha256'),
                'biology_manifest': dataset.get('biology_manifest', {}),
                'training_scope': scope['purpose'],
                'prediction_semantics': 'Conditional source severity for documented interactions; not DDI existence or patient risk.'}
    write_json(output / 'manifest.json', manifest)
    return manifest


def train_run(dataset, splits, output, *, protocol='pair', model_name='fusion', seed=17,
              epochs=60, patience=8, device='cpu', notify=None, purpose='source_label_benchmark',
              evaluate_test=True, batch_size=2048):
    import numpy as np
    import torch
    from .models import PairClassifier, tensors
    from .metrics import classification_metrics
    execution_code = code_snapshot()

    audit_splits(dataset['pairs'], splits)
    audit_predictive_features(dataset)
    if protocol not in ('pair', 'cold') or model_name not in MODEL_NAMES:
        raise ValueError('Unsupported training protocol/model.')
    if epochs < 1 or patience < 1:
        raise ValueError('Epochs and patience must be positive.')
    if batch_size < 0:
        raise ValueError('Batch size must be nonnegative; zero reproduces full-batch training.')
    expected_hash = dataset.get('content_sha256')
    if expected_hash and object_hash({k: v for k, v in dataset.items() if k != 'content_sha256'}) != expected_hash:
        raise ValueError('Dataset content hash changed.')
    if splits.get('dataset_sha256') and splits['dataset_sha256'] != expected_hash:
        raise ValueError('Split/dataset hash mismatch.')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    roles_dim = len(dataset['role_feature_names'])
    config = {'model_name': model_name, 'role_dim': roles_dim, 'fingerprint_dim': dataset['fingerprint']['bits'],
              'hidden': 128, 'embedding': 64, 'dropout': .2}
    groups = splits[protocol]
    for required in ('supervision', 'validation', 'calibration'):
        if not groups[required]:
            raise ValueError(f'Empty {protocol}/{required} split; need more eligible data.')
    graph_pairs = [dataset['pairs'][i]['ingredient_ids'] for i in groups['message']]
    index, x, roles, pairs, edges = tensors(dataset, graph_pairs, device=device)
    labels = torch.tensor([CLASSES.index(row['severity']) for row in dataset['pairs']], dtype=torch.long, device=device)
    supervision = torch.tensor(groups['supervision'], dtype=torch.long, device=device)
    class_counts = torch.bincount(labels[supervision], minlength=3).float()
    if (class_counts == 0).any():
        raise ValueError('Each source severity needs training-supervision examples.')
    if model_name == 'role_fusion':
        train_positions = sorted(set(pairs[supervision].flatten().cpu().tolist()))
        if not roles_dim or not any(float(roles[position, 0]) > 0 for position in train_positions):
            raise ValueError('Role-aware training requires observed documented roles in supervision drugs.')
    weights = torch.sqrt(len(supervision) / (3 * class_counts))
    weights /= weights.mean()
    model = PairClassifier(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    best, best_state, stale, history, optimizer_steps = -1.0, None, 0, [], 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = supervision if batch_size == 0 else supervision[torch.randperm(len(supervision), device=device)]
        width = len(order) if batch_size == 0 else batch_size
        epoch_loss = 0.0
        for offset in range(0, len(order), width):
            batch = order[offset:offset + width]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edges, pairs[batch], roles)
            loss = loss_fn(logits, labels[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            epoch_loss += float(loss.item()) * len(batch)
        model.eval()
        with torch.no_grad():
            validation_indices = groups['validation']
            val_logits = model(x, edges, pairs[validation_indices], roles).cpu().tolist()
        val_labels = labels[validation_indices].cpu().tolist()
        score = classification_metrics(softmax(val_logits), val_labels)['macro_f1']
        row = {'epoch': epoch, 'loss': epoch_loss / len(order), 'validation_macro_f1': score,
               'optimizer_steps': optimizer_steps}
        history.append(row)
        if notify:
            notify({'protocol': protocol, 'model': model_name, 'seed': seed, **row})
        if score > best + 1e-6:
            best, best_state, stale = score, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    all_labels = labels.cpu().tolist()
    temperature_ids, threshold_ids, policy_check_ids = _partition(groups['calibration'], all_labels, seed + 100,
                                                                 (.5, .25, .25))
    if not temperature_ids or not threshold_ids or not policy_check_ids:
        raise ValueError('Calibration needs nonempty fit/tune/check subsets.')
    with torch.no_grad():
        calibration_embeddings = model.encode(x, edges, roles)
        calibration_logits = model.decode(calibration_embeddings, pairs[temperature_ids]).cpu().tolist()
        threshold_logits = model.decode(calibration_embeddings, pairs[threshold_ids]).cpu().tolist()
        policy_logits = model.decode(calibration_embeddings, pairs[policy_check_ids]).cpu().tolist()
    temperature = fit_temperature(calibration_logits, labels[temperature_ids].cpu().tolist())
    policy = choose_threshold(softmax(threshold_logits, temperature), labels[threshold_ids].cpu().tolist())
    chosen_threshold = policy['threshold']
    policy['proposal'] = {'threshold': chosen_threshold, 'accepted_count': policy['accepted_count'],
                          'coverage': policy['coverage'], 'errors': policy.get('errors'),
                          'wilson_95_upper_error': policy.get('wilson_95_upper_error'),
                          'scope': 'Threshold-tuning subset before independent qualification check.'}
    check_probabilities, check_labels = softmax(policy_logits, temperature), labels[policy_check_ids].cpu().tolist()
    accepted_check = [i for i, row in enumerate(check_probabilities) if chosen_threshold is not None and max(row) >= chosen_threshold]
    errors_check = sum(max(range(3), key=lambda c: check_probabilities[i][c]) != check_labels[i] for i in accepted_check)
    upper_check = wilson_upper(errors_check, len(accepted_check)) if accepted_check else None
    policy['independent_policy_check'] = {'count': len(accepted_check), 'errors': errors_check,
                                         'wilson_95_upper_error': upper_check,
                                         'sample_count': len(policy_check_ids)}
    if chosen_threshold is not None and (len(accepted_check) < 100 or upper_check > .1):
        policy.update(threshold=None, reason='independent_policy_check_did_not_qualify', tuned_threshold=chosen_threshold)
    if policy['threshold'] is None:
        policy.update(accepted_count=0, coverage=0.0, errors=0, wilson_95_upper_error=None)
    policy.update(bookkeeping_schema='proposal-and-qualified-policy-1',
                  accepted_count_scope='qualified policy on threshold-tuning subset; zero when policy abstains')
    calibration = {'temperature': temperature, 'policy': policy, 'protocol': protocol,
                   'calibration_indices_sha256': object_hash(groups['calibration']),
                   'temperature_fit_sha256': object_hash(temperature_ids),
                   'threshold_tune_sha256': object_hash(threshold_ids),
                   'independent_policy_check_sha256': object_hash(policy_check_ids),
                   'calibration_internal_split': 'stratified 50/25/25 temperature/threshold/independent-check',
                   'scope': 'Known-source labels only; applicability to source-unknown records unvalidated.'}
    evaluation = {}
    missing_modality = {}
    with torch.no_grad():
        embeddings = model.encode(x, edges, roles)
        for role in ('validation', 'calibration', 'test', 'test_one_unseen', 'test_both_unseen'):
            if role.startswith('test') and not evaluate_test:
                continue
            if role not in groups:
                continue
            subset = groups[role]
            if not subset:
                evaluation[role] = {'count': 0, 'status': 'empty_split'}
                continue
            output_logits = model.decode(embeddings, pairs[subset]).cpu().tolist()
            y = labels[subset].cpu().tolist()
            evaluation[role] = {'before_calibration': classification_metrics(softmax(output_logits), y),
                                'after_calibration': classification_metrics(softmax(output_logits, temperature), y,
                                                                          threshold=policy['threshold'])}
            if role.startswith('test') and model_name in ('fusion', 'role_fusion'):
                empty_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
                no_graph = model(x, empty_edges, pairs[subset], roles).cpu().tolist()
                missing_modality[role] = {'without_graph_neighbors': classification_metrics(
                    softmax(no_graph, temperature), y, threshold=policy['threshold'])}
                if model_name == 'role_fusion':
                    no_roles = model(x, edges, pairs[subset], torch.zeros_like(roles)).cpu().tolist()
                    missing_modality[role]['without_documented_biology'] = classification_metrics(
                        softmax(no_roles, temperature), y, threshold=policy['threshold'])
        # Single-pair decoder latency with cached graph embeddings; embedding cost separately measured.
        for _ in range(3):
            model.decode(embeddings, pairs[:1])
        measurements = []
        for _ in range(30):
            begin = time.perf_counter()
            model.decode(embeddings, pairs[:1])
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            measurements.append((time.perf_counter() - begin) * 1000)
        begin = time.perf_counter()
        model.encode(x, edges, roles)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        embedding_ms = (time.perf_counter() - begin) * 1000
    train_drugs = sorted({i for position in groups['message'] + groups['supervision']
                          for i in dataset['pairs'][position]['ingredient_ids']})
    scope = {'purpose': purpose, 'train_drug_ids': train_drugs,
             'message_pair_keys': [dataset['pairs'][i]['source_record_id'] for i in groups['message']],
             'supervision_pair_keys_sha256': object_hash([dataset['pairs'][i]['source_record_id'] for i in groups['supervision']]),
             'split_sha256': object_hash(splits), 'supervision_count': len(supervision),
             'message_count': len(graph_pairs), 'no_confirmed_negative_labels': True,
             'execution_code': execution_code,
             'training_schedule': {'batch_size': batch_size, 'optimizer_steps': optimizer_steps,
                                    'optimizer': 'AdamW', 'learning_rate': .001,
                                    'weight_decay': .0001, 'patience_epochs': patience,
                                    'maximum_epochs': epochs}}
    config['training'] = scope['training_schedule']
    manifest = write_bundle(output, model.cpu(), config, dataset, graph_pairs, calibration, scope,
                            seed=seed, protocol=protocol)
    result = {'protocol': protocol, 'model': model_name, 'seed': seed, 'best_validation_macro_f1': best,
              'epochs_completed': len(history), 'history': history, 'class_weights': weights.cpu().tolist(),
              'batch_size': batch_size, 'optimizer_steps': optimizer_steps,
              'calibration': calibration, 'evaluation': evaluation,
              'missing_modality': missing_modality,
              'training_seconds': time.perf_counter() - started,
              'cached_pair_latency_ms': {'median': statistics.median(measurements),
                                         'p95': sorted(measurements)[int(.95 * (len(measurements) - 1))]},
              'embedding_ms': embedding_ms, 'manifest': manifest,
              'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'python': sys.version,
                           **{name: importlib.metadata.version(name) for name in ('rdkit', 'scikit-learn')}},
              'feature_coverage': {'mapped_nodes': len(dataset['nodes']),
                                    'documented_role_nodes': sum(bool(row['roles'] and row['roles'][0])
                                                                  for row in dataset['nodes']),
                                    'supervision_drugs': len(train_drugs)},
              'training_provenance': {'dataset_sha256': dataset.get('content_sha256'),
                                      'split_sha256': object_hash(splits), 'seed': seed,
                                      'source_csv_sha256': dataset.get('source_csv_sha256'),
                                      'execution_code': execution_code,
                                      'training_schedule': scope['training_schedule'],
                                      'no_negative_interaction_labels': True,
                                      'supervision_in_adjacency': False}}
    write_json(Path(output) / 'results.json', result)
    write_json(Path(output) / 'model_card.json', {'task': TASK, 'classes': list(CLASSES),
               'training_scope': purpose, 'limitations': ['Source severity conditional on documented interaction.',
               'No clinical efficacy, causality, personalized risk, or DDI-existence validation.',
               'Source-unknown pairs lack labels; distributional applicability unvalidated.'],
               'feature_coverage': result['feature_coverage'], 'provenance': result['training_provenance'],
               'calibration': calibration, 'evaluation': evaluation})
    return result


def train_all(dataset_dir, output, *, protocols=('pair', 'cold'), models=MODEL_NAMES,
              seeds=(17, 29, 43), epochs=60, patience=8, device='cpu', deployment=None, notify=None,
              batch_size=2048):
    import shutil
    dataset_dir, output = Path(dataset_dir), Path(output)
    dataset, splits = read_json(dataset_dir / 'dataset.json'), read_json(dataset_dir / 'splits.json')
    results = []
    for protocol in protocols:
        for model_name in models:
            for seed in seeds:
                path = output / protocol / model_name / str(seed)
                result = train_run(dataset, splits, path, protocol=protocol, model_name=model_name,
                                   seed=seed, epochs=epochs, patience=patience, device=device, notify=notify,
                                   batch_size=batch_size)
                # Batch scheduling is part of the run provenance, not an architecture change.
                results.append({'path': str(path), **{k: result[k] for k in ('protocol', 'model', 'seed',
                               'best_validation_macro_f1', 'evaluation', 'training_seconds')}})
                write_json(output / 'runs.json', {'runs': results, 'status': 'in_progress'})
    aggregate = []
    for protocol in protocols:
        for model_name in models:
            rows = [r for r in results if r['protocol'] == protocol and r['model'] == model_name]
            vals = [r['best_validation_macro_f1'] for r in rows]
            aggregate.append({'protocol': protocol, 'model': model_name, 'seeds': list(seeds),
                              'validation_macro_f1_mean': statistics.mean(vals),
                              'validation_macro_f1_sd': statistics.stdev(vals) if len(vals) > 1 else 0})
            aggregate[-1]['test_conditions'] = {}
            for condition in ('test', 'test_one_unseen', 'test_both_unseen'):
                scored = [row['evaluation'][condition]['after_calibration'] for row in rows
                          if condition in row['evaluation'] and row['evaluation'][condition].get('count', 1)]
                if not scored:
                    continue
                metrics = {}
                for metric in ('macro_f1', 'high_pr_auc', 'nll', 'brier', 'ece'):
                    values = [r[metric] for r in scored if r.get(metric) is not None]
                    metrics[metric] = {'mean': statistics.mean(values),
                                       'sd': statistics.stdev(values) if len(values) > 1 else 0} if values else None
                for metric in ('coverage', 'error_rate'):
                    values = [r['selective'][metric] for r in scored if r['selective'].get(metric) is not None]
                    metrics['selective_' + metric] = {'mean': statistics.mean(values),
                        'sd': statistics.stdev(values) if len(values) > 1 else 0} if values else None
                aggregate[-1]['test_conditions'][condition] = metrics
    selected = None
    pair_rows = [row for row in aggregate if row['protocol'] == 'pair']
    if pair_rows:
        winner = max(pair_rows, key=lambda r: r['validation_macro_f1_mean'])
        contenders = [r for r in results if r['protocol'] == 'pair' and r['model'] == winner['model']]
        selected = min(contenders, key=lambda r: abs(r['best_validation_macro_f1'] - winner['validation_macro_f1_mean']))
        if deployment:
            # Copy an entire integrity-checked bundle; model choice never consults test scores.
            shutil.copytree(selected['path'], deployment, dirs_exist_ok=True)
            write_json(Path(deployment) / 'selection.json', {'criterion': 'validation mean across seeds; representative seed',
                                                           'selected': selected['path'], 'aggregate': aggregate})
    summary = {'status': 'complete', 'runs': results, 'aggregate': aggregate,
               'selected_path': selected['path'] if selected else None,
               'selection_uses_test_data': False,
               'scope': 'Automatic held-out source-label benchmark; no clinical validation or human study.'}
    write_json(output / 'runs.json', summary)
    write_json(output / 'summary.json', summary)
    return summary
