"""Research-only recognition stress test on masked, known-source TEST pairs.

Reference identities generate observations and score results; only observation
text enters normalization. Model classes mean conditional source severity, never
interaction existence or patient risk. The application unknown-only gate is not
used or changed by this separate research runner.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adr_system.ml import CLASSES
from adr_system.ml.io import file_hash, object_hash, read_json, write_json

VARIANTS = ('clean', 'single_character_error', 'mixed_corruption')


def pair_key(ids) -> str:
    return '|'.join(sorted(ids))


def audit_holdout(source: dict, splits: dict, scope: dict | None = None) -> dict:
    """Check canonical TEST identity against both message and supervision roles."""
    pairs, roles = source['pairs'], splits['pair']
    keys = []
    for row in pairs:
        ids = row['ingredient_ids']
        if (len(ids) != 2 or len(set(ids)) != 2 or row['severity'] not in CLASSES
                or row['source_record_id'] != pair_key(ids)):
            raise ValueError('Noncanonical or invalid labeled source pair.')
        keys.append(row['source_record_id'])
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate or reversed source pair.')
    seen = set()
    for indices in roles.values():
        if (len(set(indices)) != len(indices)
                or any(not isinstance(i, int) or not 0 <= i < len(pairs) for i in indices)
                or seen.intersection(indices)):
            raise ValueError('Pair split leakage or invalid split indices.')
        seen.update(indices)
    if seen != set(range(len(pairs))):
        raise ValueError('Unaccounted source pairs in pair split.')
    test_keys = {keys[i] for i in roles['test']}
    training_keys = {keys[i] for role in ('message', 'supervision') for i in roles[role]}
    if test_keys & training_keys:
        raise ValueError('True query pair occurs in message or supervision data.')
    if scope is not None:
        expected_message = [keys[i] for i in roles['message']]
        expected_supervision = [keys[i] for i in roles['supervision']]
        if (scope.get('purpose') != 'source_label_benchmark'
                or set(scope.get('message_pair_keys', [])) != set(expected_message)
                or scope.get('supervision_pair_keys_sha256') != object_hash(expected_supervision)
                or scope.get('split_sha256') != object_hash(splits)
                or test_keys.intersection(scope.get('message_pair_keys', []))):
            raise ValueError('Model message/supervision provenance does not match the frozen pair split.')
    return {'passed': True, 'test_pairs': len(test_keys), 'message_pairs': len(roles['message']),
            'supervision_pairs': len(roles['supervision']), 'true_query_training_overlap': 0,
            'canonical_source_record_ids': True}


def corrupt(name: str, variant: str, rng: random.Random) -> str:
    if variant == 'clean':
        return name
    position = rng.randrange(len(name))
    if variant == 'single_character_error':
        replacement = 'l' if name[position].casefold() != 'l' else 'i'
        return name[:position] + replacement + name[position + 1:]
    operation = rng.choice(('substitute', 'delete', 'transpose', 'ocr_confusion'))
    if operation == 'delete' and len(name) > 1:
        return name[:position] + name[position + 1:]
    if operation == 'transpose' and len(name) > 1:
        position = min(position, len(name) - 2)
        value = name[:position] + name[position + 1] + name[position] + name[position + 2:]
        if value != name:
            return value
    if operation == 'ocr_confusion':
        value = name.translate(str.maketrans({'o': '0', 'i': '1', 's': '5', 'l': '1'}))
        if value != name:
            return value
    replacement = 'x' if name[position].casefold() != 'x' else 'z'
    return name[:position] + replacement + name[position + 1:]


def make_dataset(source: dict, splits: dict, *, count: int = 100, seed: int = 1729,
                 scope: dict | None = None) -> dict:
    if count < 1:
        raise ValueError('count must be positive')
    audit = audit_holdout(source, splits, scope)
    nodes = {row['ingredient_id']: row for row in source['nodes']}
    eligible, missing = [], 0
    for index in splits['pair']['test']:
        row = source['pairs'][index]
        if not all(ident in nodes and nodes[ident].get('name', '').strip() for ident in row['ingredient_ids']):
            missing += 1
        else:
            eligible.append(index)
    if count > len(eligible):
        raise ValueError(f'Requested {count} pairs; only {len(eligible)} TEST pairs have reference molecule names.')
    rng = random.Random(seed)
    selected = rng.sample(sorted(eligible), count)
    cases = []
    for group, index in enumerate(selected):
        row = source['pairs'][index]
        ids = sorted(row['ingredient_ids'])
        names = [nodes[ident]['name'] for ident in ids]
        reference = {'ingredient_ids': ids, 'source_record_id': row['source_record_id'],
                     'source_severity': row['severity']}
        for variant in VARIANTS:
            cases.append({'id': f'MREC-{group:04d}-{variant}', 'group_id': f'M{group:04d}',
                          'split': 'test', 'input_kind': variant,
                          'entries': [{'text': corrupt(name, variant, rng)} for name in names],
                          'reference': reference.copy()})
    return {'schema': 'model-recognition-stress-1', 'kind': 'masked_known_source_pair_test',
            'seed': seed, 'base_pairs': count, 'variants': list(VARIANTS), 'cases': cases,
            'source_dataset_sha256': source.get('content_sha256'), 'split_sha256': object_hash(splits),
            'holdout_audit': audit,
            'generation_coverage': {'test_pairs': audit['test_pairs'], 'eligible_test_pairs': len(eligible),
                                    'missing_reference_molecule_or_name_pairs': missing,
                                    'frozen_dataset_exclusions': source.get('exclusions', {})},
            'scope': 'Synthetic two-entry recognition; conditional source severity on known labeled TEST pairs.',
            'label_use': 'Reference names generate text; reference identities and labels are used only for scoring.'}


def retained_options(medication) -> list[tuple[str, ...]]:
    if medication.ingredient_ids:
        return [tuple(sorted(medication.ingredient_ids))]
    # Preserve lexical order: no label, molecule availability, or gold identity reranking.
    return list(dict.fromkeys(tuple(sorted(c.ingredient_ids)) for c in medication.candidates[:5]))


def _prediction(row: dict | None, reason='prediction_unavailable') -> dict:
    if row is None:
        return {'available': False, 'accepted': False, 'predicted_source_severity': None,
                'probabilities': {}, 'abstention_reason': reason}
    probabilities = row.get('probabilities', {})
    values = [probabilities.get(label) for label in CLASSES]
    valid = (all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                 for value in values) and abs(sum(values) - 1) < 1e-4)
    prediction = CLASSES[max(range(len(CLASSES)), key=lambda i: values[i])] if valid else None
    if prediction != row.get('predicted_severity'):
        valid, prediction = False, None
    return {'available': valid, 'accepted': bool(valid and row.get('accepted')),
            'predicted_source_severity': prediction, 'probabilities': probabilities if valid else {},
            'abstention_reason': row.get('abstention_reason') if valid else
                                 row.get('abstention_reason') or 'prediction_unavailable'}


def evaluate_case(case: dict, *, model_ids: set[str], predictor, model_dir=None,
                  normalizer=None, source_checker=None) -> dict:
    """Pure scoring path with an injected predictor; no optional ML imports here."""
    if normalizer is None:
        from adr_system.normalization import normalise_medication
        normalizer = normalise_medication
    if source_checker is None:
        from adr_system.knowledge import pair_state
        source_checker = pair_state
    if len(case['entries']) != 2:
        raise ValueError('Recognition stress cases must contain exactly two entries.')
    # Only strings cross this boundary. Gold identities never enter normalization.
    medications = [normalizer(entry['text']) for entry in case['entries']]
    options = [retained_options(medication) for medication in medications]
    combinations = list(itertools.product(*options))
    valid_pairs = {tuple(sorted((a[0], b[0]))) for a, b in combinations
                   if len(a) == len(b) == 1 and a != b}
    # Source state is reduced to existence only. Its severity is never an input,
    # a candidate ranking signal, or a substitute for a model prediction.
    documented = {pair: source_checker(*pair) in (*CLASSES, 'unknown') for pair in valid_pairs}
    usable = [tuple(sorted((a[0], b[0]))) for a, b in combinations
              if len(a) == len(b) == 1 and a != b and a[0] in model_ids and b[0] in model_ids
              and documented[tuple(sorted((a[0], b[0])))]]
    unique_pairs = sorted(set(usable))
    predictions = predictor([list(pair) for pair in unique_pairs], model_dir) if unique_pairs else []
    if len(predictions) != len(unique_pairs):
        raise ValueError('Predictor returned a different number of candidate pairs.')
    by_pair = {}
    for pair, prediction in zip(unique_pairs, predictions):
        if prediction.get('source_record_id', pair_key(pair)) != pair_key(pair):
            raise ValueError('Predictor returned a mismatched source_record_id.')
        by_pair[pair] = _prediction(prediction)
    outcomes = []
    for a, b in combinations:
        ids = tuple(sorted(a + b))
        valid_pair = len(a) == len(b) == 1 and a != b
        missing = sorted(set(ids) - model_ids)
        source_present = documented.get(ids) if valid_pair else None
        outcome = by_pair.get(ids) if valid_pair and source_present and not missing else None
        reason = ('invalid_single_ingredient_pair' if not valid_pair else
                  'no_source_record' if not source_present else
                  'molecular_mapping_unavailable' if missing else 'prediction_unavailable')
        outcomes.append({'ingredient_ids': list(ids), 'missing_molecule_ids': missing,
                         'valid_single_ingredient_pair': valid_pair,
                         'source_record_present': source_present,
                         **(_prediction(None, reason) if outcome is None else outcome)})
    top_ids = tuple(sorted(options[0][0] + options[1][0])) if all(options) else ()
    top_valid = bool(all(options) and len(options[0][0]) == len(options[1][0]) == 1
                     and options[0][0] != options[1][0])
    top_missing = sorted(set(top_ids) - model_ids)
    top_reason = ('no_retained_candidate_pair' if not all(options) else
                  'invalid_single_ingredient_pair' if not top_valid else
                  'no_source_record' if not documented.get(top_ids) else
                  'molecular_mapping_unavailable' if top_missing else 'prediction_unavailable')
    top = by_pair.get(top_ids, _prediction(None, top_reason)) if top_valid else _prediction(None, top_reason)
    available = [outcome for outcome in outcomes if outcome['available']]
    classes = sorted({outcome['predicted_source_severity'] for outcome in available})
    stable = bool(outcomes) and len(available) == len(outcomes) and len(classes) == 1
    status = 'stable' if stable else 'conditional' if available else 'unusable'
    missing_ids = sorted({ident for outcome in outcomes for ident in outcome['missing_molecule_ids']})
    # Reference fields are first consulted AFTER candidate generation and prediction.
    reference = case['reference']
    gold = reference['ingredient_ids']
    hits = [tuple([ident]) in choices for ident, choices in zip(gold, options)]
    top_hits = [bool(choices) and choices[0] == (ident,) for ident, choices in zip(gold, options)]
    return {'case_id': case['id'], 'group_id': case['group_id'], 'split': 'test',
            'input_kind': case['input_kind'], 'source_record_id': reference['source_record_id'],
            'source_severity': reference['source_severity'], 'reference_ingredient_ids': gold,
            'reference_missing_model_molecule_ids': sorted(set(gold) - model_ids),
            'observed_text': [entry['text'] for entry in case['entries']],
            'retained_candidates': [[list(ids) for ids in choices] for choices in options],
            'candidate_identity_hits': sum(hits), 'top1_identity_hits': sum(top_hits),
            'candidate_true_pair_covered': all(hits), 'top1_true_pair_identity': all(top_hits),
            'no_candidate_entries': sum(not choices for choices in options),
            'top1': {'ingredient_ids': list(top_ids), 'missing_molecule_ids': top_missing,
                     'source_record_present': documented.get(top_ids) if top_valid else None, **top},
            'candidate_set': {'status': status,
                              'predicted_source_severity': classes[0] if stable else None,
                              'accepted': stable and all(outcome['accepted'] for outcome in available),
                              'combinations': len(outcomes),
                              'mapped_valid_pairs': sum(o['valid_single_ingredient_pair'] and not o['missing_molecule_ids'] for o in outcomes),
                              'documented_pairs': sum(o['source_record_present'] is True for o in outcomes),
                              'no_source_record_pairs': sum(o['source_record_present'] is False for o in outcomes),
                              'eligible_documented_mapped_pairs': len(usable),
                              'model_available_pairs': len(available), 'model_classes': classes,
                              'missing_molecule_ids': missing_ids,
                              'missing_molecule_pairs': sum(bool(o['missing_molecule_ids']) for o in outcomes),
                              'outcomes': outcomes}}


def rate(numerator: int, denominator: int) -> dict:
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


def _selective(rows: list[dict], method: str) -> dict:
    accepted = [row for row in rows if row[method]['accepted']]
    errors = sum(row[method]['predicted_source_severity'] != row['source_severity'] for row in accepted)
    return {'accepted_coverage': rate(len(accepted), len(rows)), 'accepted_error': rate(errors, len(accepted)),
            'abstention': rate(len(rows) - len(accepted), len(rows)),
            'per_class': {label: {'accepted_coverage': rate(
                sum(row[method]['accepted'] for row in rows if row['source_severity'] == label),
                sum(row['source_severity'] == label for row in rows)),
                'accepted_error': rate(sum(row[method]['predicted_source_severity'] != label for row in accepted
                                           if row['source_severity'] == label),
                                       sum(row['source_severity'] == label for row in accepted))} for label in CLASSES}}


def summarize(rows: list[dict], metric_fn) -> list[dict]:
    """Compute metrics only on model-available top-1 / stable candidate-set cases."""
    output = []
    for variant in VARIANTS:
        part = [row for row in rows if row['input_kind'] == variant]
        top_eligible = [row for row in part if row['top1']['available']]
        stable = [row for row in part if row['candidate_set']['status'] == 'stable']
        top_missing_ids = sorted({ident for row in part for ident in row['top1']['missing_molecule_ids']})
        set_missing_ids = sorted({ident for row in part for ident in row['candidate_set']['missing_molecule_ids']})
        top_metrics = metric_fn([[row['top1']['probabilities'][label] for label in CLASSES] for row in top_eligible],
                                [CLASSES.index(row['source_severity']) for row in top_eligible])
        # Hard classes support classification scores; they do not define a set probability.
        stable_metrics = metric_fn([[int(label == row['candidate_set']['predicted_source_severity'])
                                     for label in CLASSES] for row in stable],
                                   [CLASSES.index(row['source_severity']) for row in stable])
        def classification(metrics, eligible, *, probability_score):
            return {'eligible_count': len(eligible), 'total_case_count': len(part),
                    'source_class_counts': dict(Counter(row['source_severity'] for row in eligible)),
                    'macro_f1': metrics['macro_f1'], 'per_class': metrics['per_class'],
                    'high_pr_auc': metrics['high_pr_auc'] if probability_score else None,
                    'high_pr_auc_positive_count': sum(row['source_severity'] == 'high' for row in eligible),
                    'high_pr_auc_negative_count': sum(row['source_severity'] != 'high' for row in eligible),
                    'high_pr_auc_reason': None if probability_score else 'No candidate-set probability aggregation is defined.'}
        output.append({'input_kind': variant, 'cases': len(part), 'entries': 2 * len(part),
                       'candidate_true_identity_coverage': rate(sum(r['candidate_identity_hits'] for r in part), 2 * len(part)),
                       'top1_identity_rate': rate(sum(r['top1_identity_hits'] for r in part), 2 * len(part)),
                       'candidate_true_pair_coverage': rate(sum(r['candidate_true_pair_covered'] for r in part), len(part)),
                       'top1_pair_identity_rate': rate(sum(r['top1_true_pair_identity'] for r in part), len(part)),
                       'no_candidate_entries': sum(r['no_candidate_entries'] for r in part),
                       'reference_missing_model_molecule_cases': sum(bool(r['reference_missing_model_molecule_ids']) for r in part),
                       'top1': {'model_availability': rate(len(top_eligible), len(part)),
                                'missing_molecule_cases': sum(bool(r['top1']['missing_molecule_ids']) for r in part),
                                'missing_molecule_count': len(top_missing_ids), 'missing_molecule_ids': top_missing_ids,
                                'no_source_record_cases': sum(r['top1']['source_record_present'] is False for r in part),
                                'raw_model_argmax': classification(top_metrics, top_eligible, probability_score=True),
                                'calibrated_decisions': _selective(part, 'top1'),
                                'abstention_reasons': dict(Counter(r['top1']['abstention_reason'] for r in part
                                                                  if not r['top1']['accepted']))},
                       'candidate_set': {'status_counts': dict(Counter(r['candidate_set']['status'] for r in part)),
                                         'model_availability': rate(sum(r['candidate_set']['model_available_pairs'] > 0 for r in part), len(part)),
                                         'stable_class_coverage': rate(len(stable), len(part)),
                                         'missing_molecule_pairs': sum(r['candidate_set']['missing_molecule_pairs'] for r in part),
                                         'missing_molecule_cases': sum(bool(r['candidate_set']['missing_molecule_ids']) for r in part),
                                         'missing_molecule_count': len(set_missing_ids), 'missing_molecule_ids': set_missing_ids,
                                         'no_source_record_pairs': sum(r['candidate_set']['no_source_record_pairs'] for r in part),
                                         'no_source_record_cases': sum(r['candidate_set']['no_source_record_pairs'] > 0 for r in part),
                                         'raw_stable_model_class': classification(stable_metrics, stable, probability_score=False),
                                         'calibrated_decisions': _selective(part, 'candidate_set')}})
    return output


def evaluate(dataset_dir: Path, model_dir: Path, output: Path, *, count=100, seed=1729) -> dict:
    # Optional ML imports stay here so --help and injected-predictor tests use the core environment.
    from adr_system.ml.inference import bundle_manifest, predict_pairs
    from adr_system.ml.metrics import classification_metrics

    input_hashes = {name: file_hash(dataset_dir / name) for name in ('dataset.json', 'splits.json')}
    source = read_json(dataset_dir / 'dataset.json')
    splits = read_json(dataset_dir / 'splits.json')
    status = bundle_manifest(model_dir)
    if status['status'] != 'available' or not status.get('runtime_available'):
        raise ValueError(f'Model bundle/runtime unavailable: {status}')
    if (status.get('protocol') != 'pair' or status.get('dataset_sha256') != source.get('content_sha256')
            or splits.get('dataset_sha256') != source.get('content_sha256')):
        raise ValueError('This runner requires the frozen dataset and its matching pair-protocol model.')
    source_content = {key: value for key, value in source.items() if key != 'content_sha256'}
    if object_hash(source_content) != source['content_sha256']:
        raise ValueError('Frozen dataset content hash mismatch.')
    scope = read_json(model_dir / 'training_scope.json')
    dataset = make_dataset(source, splits, count=count, seed=seed, scope=scope)
    model_ids = {row['ingredient_id'] for row in read_json(model_dir / 'features.json')['nodes']}
    cache = {}
    def cached_predictor(pairs, directory):
        unseen = [pair for pair in pairs if pair_key(pair) not in cache]
        for pair, result in zip(unseen, predict_pairs(unseen, directory)):
            cache[pair_key(pair)] = result
        return [cache[pair_key(pair)] for pair in pairs]
    rows = [evaluate_case(case, model_ids=model_ids, predictor=cached_predictor, model_dir=model_dir)
            for case in dataset['cases']]
    # Never write a report mixing predictions from bundles modified mid-run.
    if bundle_manifest(model_dir) != status:
        raise ValueError('Model bundle changed during evaluation; rerun against a frozen bundle.')
    if any(file_hash(dataset_dir / name) != digest for name, digest in input_hashes.items()):
        raise ValueError('Dataset or splits changed during evaluation; rerun against frozen inputs.')
    calibration = read_json(model_dir / 'calibration.json')
    summary = {'execution_status': 'executed', 'dataset_kind': dataset['kind'], 'seed': seed,
               'independent_test_pairs': count, 'cases_evaluated': len(rows),
               'source_dataset_sha256': source['content_sha256'],
               'source_dataset_file_sha256': input_hashes['dataset.json'],
               'splits_file_sha256': input_hashes['splits.json'],
               'benchmark_dataset_sha256': object_hash(dataset), 'model': status,
               'runner_sha256': file_hash(__file__), 'holdout_audit': dataset['holdout_audit'],
               'generation_coverage': dataset['generation_coverage'],
               'calibration_threshold': calibration['policy']['threshold'], 'comparison': summarize(rows, classification_metrics),
               'semantics': {'raw_argmax': 'Research stress measure, including model predictions rejected by calibration.',
                             'candidate_stability': 'Every retained candidate pair must be valid, documented, mapped, and have the same available model class.',
                             'source_gate': 'Source state checks documented-record existence only; source severity never enters predictor inputs.',
                             'calibrated_candidate_acceptance': 'Stable class and every retained pair individually accepted by the model policy.',
                             'probabilities': 'Model probabilities only; lexical scores are never converted or averaged.',
                             'candidate_high_pr_auc': 'Undefined without a candidate-set probability model; top-1 uses actual model high-class probabilities.'},
               'limitations': ['Synthetic recognition stress is not clinical validation.',
                              'Known source-label TEST queries are masked for this research API; application unknown-only eligibility is unchanged.',
                              'Candidate model outputs assume identities and documented interaction; they cannot establish interaction existence.',
                              'Stable model classes can be wrong when true identities are absent from candidates.',
                              'Missing mappings and all abstentions remain in coverage denominators.',
                              'Labels describe source severity, not patient risk.']}
    write_json(output / 'dataset.json', dataset)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'case_results.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    write_json(output / 'summary.json', summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/ml/dataset')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'artifacts/ml/deployment')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/hybrid/model-recognition')
    parser.add_argument('--count', type=int, default=100, help='Independent held-out TEST pairs; each has three text variants.')
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    summary = evaluate(args.dataset, args.model_dir, args.output, count=args.count, seed=args.seed)
    print(json.dumps({'execution_status': summary['execution_status'], 'test_pairs': summary['independent_test_pairs'],
                      'cases': summary['cases_evaluated'], 'output': str(args.output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
