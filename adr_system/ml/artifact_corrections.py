"""Explicit post-run correction of calibration bookkeeping; numerical inference unchanged."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from .io import read_json, write_json, file_hash, object_hash


def correct_policy_bookkeeping(artifacts_dir):
    directory = Path(artifacts_dir)
    manifest = read_json(directory / 'manifest.json')
    calibration = read_json(directory / 'calibration.json')
    policy = calibration['policy']
    if policy.get('bookkeeping_schema') == 'proposal-and-qualified-policy-1':
        return {'status': 'already_corrected', 'model_version': manifest['model_version']}
    original_calibration = copy.deepcopy(calibration)
    original_manifest = copy.deepcopy(manifest)
    policy['proposal'] = {'threshold': policy.get('threshold') if policy.get('threshold') is not None else policy.get('tuned_threshold'),
                          'accepted_count': policy.get('accepted_count', 0), 'coverage': policy.get('coverage', 0),
                          'errors': policy.get('errors'), 'wilson_95_upper_error': policy.get('wilson_95_upper_error'),
                          'scope': 'Threshold-tuning subset before independent qualification check.'}
    if policy['threshold'] is None:
        policy.update(accepted_count=0, coverage=0.0, errors=0, wilson_95_upper_error=None)
    policy.update(bookkeeping_schema='proposal-and-qualified-policy-1',
                  accepted_count_scope='qualified policy on threshold-tuning subset; zero when policy abstains')
    write_json(directory / 'calibration_raw_before_bookkeeping.json', original_calibration)
    write_json(directory / 'manifest_raw_before_bookkeeping.json', original_manifest)
    write_json(directory / 'calibration.json', calibration)
    manifest['files']['calibration.json'] = file_hash(directory / 'calibration.json')
    manifest['model_version'] = object_hash(manifest['files'])[:20]
    manifest['post_run_correction'] = {'reason': 'Separate proposed threshold statistics from qualified abstention policy.',
                                        'numerical_predictions_or_acceptance_changed': False,
                                        'previous_model_version': original_manifest['model_version']}
    write_json(directory / 'manifest.json', manifest)
    for name in ('results.json', 'model_card.json'):
        path = directory / name
        if path.exists():
            payload = read_json(path)
            payload['calibration'] = calibration
            if 'manifest' in payload:
                payload['manifest'] = manifest
            payload['post_run_correction'] = manifest['post_run_correction']
            write_json(path, payload)
    return {'status': 'corrected', 'model_version': manifest['model_version'],
            'previous_model_version': original_manifest['model_version'],
            'qualified_threshold': policy['threshold'], 'qualified_coverage': policy['coverage']}


def correct_training_directory(training_dir, deployment=None):
    training_dir = Path(training_dir)
    summary = read_json(training_dir / 'summary.json')
    if summary['status'] != 'complete':
        raise ValueError('Finish training before correcting immutable run artifacts.')
    original_summary = copy.deepcopy(summary)
    corrections = [{'path': run['path'], **correct_policy_bookkeeping(run['path'])} for run in summary['runs']]
    if any(row['status'] == 'corrected' for row in corrections):
        write_json(training_dir / 'summary_raw_before_bookkeeping.json', original_summary)
    # The summary embeds run results, so update its calibration and manifest too.
    # Test metrics already use the qualified threshold and must remain unchanged.
    summary['runs'] = [{'path': run['path'], **read_json(Path(run['path']) / 'results.json')}
                       for run in summary['runs']]
    summary['post_run_correction'] = {'reason': 'Proposal and qualified calibration statistics separated.',
                                      'numerical_model_selection_changed': False}
    write_json(training_dir / 'summary.json', summary)
    write_json(training_dir / 'runs.json', summary)
    write_json(training_dir / 'bookkeeping_corrections.json', {'status': 'executed', 'corrections': corrections,
                                                              'numerical_model_selection_changed': False})
    if deployment:
        shutil.copytree(summary['selected_path'], deployment, dirs_exist_ok=True)
        write_json(Path(deployment) / 'selection.json', {'criterion': 'validation mean across seeds; representative seed',
                                                        'selected': summary['selected_path'], 'aggregate': summary['aggregate']})
    return corrections
