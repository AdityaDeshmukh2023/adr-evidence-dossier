from __future__ import annotations

from pathlib import Path
import uuid

from adr_system.ml.artifact_corrections import correct_training_directory
from adr_system.ml.io import file_hash, read_json, write_json


def test_failed_policy_preserves_proposal_and_zeros_deployed_summary():
    root = Path(__file__).resolve().parents[1] / 'artifacts/ml/tests' / uuid.uuid4().hex
    run = root / 'pair/fingerprint/17'
    run.mkdir(parents=True)
    calibration = {'temperature': 1.3, 'policy': {
        'threshold': None, 'tuned_threshold': .8, 'accepted_count': 110,
        'coverage': .55, 'errors': 3, 'wilson_95_upper_error': .077,
        'independent_policy_check': {'qualified': False}}}
    write_json(run / 'calibration.json', calibration)
    manifest = {'model_version': 'before', 'files': {'calibration.json': file_hash(run / 'calibration.json')}}
    write_json(run / 'manifest.json', manifest)
    numerical = {'test': {'selective': {'coverage': 0, 'accepted_count': 0}}}
    result = {'calibration': calibration, 'manifest': manifest, 'evaluation': numerical}
    write_json(run / 'results.json', result)
    write_json(run / 'model_card.json', {'calibration': calibration, 'evaluation': numerical})
    write_json(root / 'summary.json', {'status': 'complete', 'runs': [{'path': str(run), **result}],
                                      'selected_path': str(run), 'aggregate': []})

    corrections = correct_training_directory(root, root / 'deployment')
    policy = read_json(run / 'calibration.json')['policy']
    assert policy['threshold'] is None
    assert policy['accepted_count'] == 0
    assert policy['coverage'] == 0
    assert policy['wilson_95_upper_error'] is None
    assert policy['proposal']['threshold'] == .8
    assert policy['proposal']['accepted_count'] == 110
    corrected = read_json(root / 'summary.json')['runs'][0]
    assert corrected['path'] == str(run)
    assert corrected['calibration']['policy'] == policy
    assert corrected['evaluation'] == numerical
    assert corrected['manifest']['files']['calibration.json'] == file_hash(run / 'calibration.json')
    assert corrections[0]['model_version'] != 'before'
    assert read_json(root / 'deployment/calibration.json')['policy'] == policy
    assert read_json(root / 'summary_raw_before_bookkeeping.json')['runs'][0]['calibration'] == calibration
    assert correct_training_directory(root)[0]['status'] == 'already_corrected'
