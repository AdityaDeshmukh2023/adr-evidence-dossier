import copy
from dataclasses import replace

import pytest

from adr_system.models import Candidate, Medication
from adr_system.normalization import normalise_medication
from adr_system.research import source_reference
from adr_system.terminology import exact
from scripts.hybrid_benchmark import (aggregate, evaluate_medications, paired_bootstrap,
                                     score_findings, synthetic_dataset, validate_dataset)
from scripts.hybrid_ocr_benchmark import align_for_scoring, synthetic_images, validate_image_dataset


def test_hybrid_groups_are_repeatable_and_variants_cannot_leak():
    first = synthetic_dataset(10, seed=23)
    assert first == synthetic_dataset(10, seed=23)
    assert len(first['cases']) == 30
    assert {c['size'] for c in first['cases']} == {2, 5, 10, 20, 30}
    validate_dataset(first)
    changed = copy.deepcopy(first)
    changed['cases'][1]['split'] = 'test'
    with pytest.raises(ValueError, match='leak'):
        validate_dataset(changed)


def test_source_potential_findings_do_not_resolve_the_uncertain_identity():
    pytest.importorskip('adr_system.uncertainty')
    entry_a = {'text': 'warfarln', 'gold_names': ['warfarin']}
    entry_b = {'text': 'aspirin', 'gold_names': ['aspirin']}
    medications = [replace(normalise_medication(entry_a['text']), mention_id='m1'),
                   replace(normalise_medication(entry_b['text']), mention_id='m2')]
    assert not medications[0].ingredient_ids
    case = {'id': 'case', 'group_id': 'group', 'split': 'test', 'input_kind': 'corrupted',
            'size': 2, 'entries': [entry_a, entry_b],
            'gold_findings': source_reference([entry_a, entry_b])}
    rows = {r['method']: r for r in evaluate_medications(case, medications)}
    assert rows['resolved_only']['tp'] == 0
    assert rows['candidate_propagation']['tp'] >= 1
    assert rows['candidate_propagation']['confirmed_tp'] == 0
    assert rows['candidate_propagation']['conditional_findings'] >= 1
    assert not medications[0].ingredient_ids  # evaluating never changes the source identity


def test_metrics_do_not_call_extra_source_findings_clinical_false_positives():
    gold = {(('a', 'b'), 'high')}
    predicted = gold | {(('a', 'c'), 'moderate')}
    metrics = score_findings(predicted, gold)
    assert metrics == {'tp': 1, 'fp': 1, 'fn': 0, 'high_tp': 1, 'high_total': 1,
                       'finding_total': 1, 'predicted_total': 2}


def test_bootstrap_uses_test_groups_and_reports_insufficient_groups():
    rows = []
    for group, split in [('one', 'development'), ('two', 'test')]:
        for method in ('resolved_only', 'candidate_propagation'):
            rows.append({'group_id': group, 'split': split, 'input_kind': 'corrupted',
                         'method': method, 'high_tp': 1, 'high_total': 2})
    assert paired_bootstrap(rows)['ci95'] is None
    for row in list(rows):
        if row['split'] == 'test':
            rows.append({**row, 'group_id': 'three'})
    interval = paired_bootstrap(rows, repeats=50)
    assert interval['test_groups'] == 2
    assert interval['difference'] == 0
    assert interval['ci95'] == [0, 0]


def test_geometry_alignment_does_not_use_ocr_words_to_choose_gold():
    reference = [{'text': 'aspirin', 'gold_names': ['aspirin'], 'gold_bbox': [10, 20, 300, 70]}]
    wrong = replace(normalise_medication('metformin'), mention_id='m1', bbox=(10, 20, 300, 70))
    mapping, metrics = align_for_scoring([wrong], reference)
    assert mapping['m1'] == exact('aspirin')[0].ingredient_ids
    assert wrong.ingredient_ids != mapping['m1']
    assert metrics['boundary_matches'] == 1
    assert metrics['bbox_iou_sum'] == 1


def test_generated_image_manifest_preserves_groups_and_detects_changed_bytes():
    from pathlib import Path
    from uuid import uuid4
    output = Path(__file__).resolve().parents[1] / 'artifacts/hybrid/test_images' / uuid4().hex
    dataset = synthetic_images(output, groups=2, seed=11)
    assert len(dataset['cases']) == 6
    validate_image_dataset(dataset)
    image = dataset['cases'][0]['image']
    Path(image).write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes'):
        validate_image_dataset(dataset)


def test_missing_image_entries_stay_in_candidate_coverage_denominator():
    pytest.importorskip('adr_system.uncertainty')
    entries = [{'text': 'warfarin', 'gold_names': ['warfarin']},
               {'text': 'aspirin', 'gold_names': ['aspirin']}]
    medication = replace(normalise_medication('aspirin'), mention_id='m1')
    case = {'id': 'image', 'group_id': 'group', 'split': 'test', 'input_kind': 'clean',
            'size': 2, 'entries': entries, 'gold_findings': source_reference(entries)}
    rows = evaluate_medications(case, [medication],
                               gold_by_mention={'m1': exact('aspirin')[0].ingredient_ids}, gold_entry_count=2)
    assert rows[0]['candidate_hits'] == 1
    assert aggregate([rows[0]])['candidate_coverage'] == .5


def test_stable_source_predicate_can_be_wrong_when_true_identity_is_missing():
    pytest.importorskip('adr_system.uncertainty')
    entries = [{'text': 'unclear', 'gold_names': ['metformin']},
               {'text': 'aspirin', 'gold_names': ['aspirin']}]
    warfarin = exact('warfarin')[0]
    unclear = Medication('unclear', None, mention_id='m1', candidates=(warfarin,), status='ambiguous')
    partner = replace(normalise_medication('aspirin'), mention_id='m2')
    case = {'id': 'candidate-failure', 'group_id': 'group', 'split': 'test',
            'input_kind': 'mixed_corruption', 'size': 2, 'entries': entries,
            'gold_findings': source_reference(entries)}
    proposed = evaluate_medications(case, [unclear, partner])[-1]
    assert proposed['stable_predicates'] >= 1
    assert proposed['incorrect_stable_predicates'] >= 1
    assert proposed['candidate_hits'] == 1


def test_stable_severity_does_not_assert_an_exact_candidate_pair_identity():
    from adr_system.uncertainty import candidate_assessments
    partner = replace(normalise_medication('aspirin'), mention_id='m2')
    unresolved = Medication('unclear', None, mention_id='m1',
                            candidates=(exact('warfarin')[0], exact('apixaban')[0]),
                            status='ambiguous')
    assessment = candidate_assessments([unresolved, partner])[0]
    assert assessment['status'] == 'stable'
    assert assessment['candidate_counts'] == [2, 1]
    assert assessment['stable_states'] == ['high']
    assert assessment['stable_record_ids'] == []
    assert not unresolved.ingredient_ids


def test_paper_bundle_preserves_historical_outputs_and_rejects_overwrite():
    import json
    from pathlib import Path
    from uuid import uuid4
    from scripts.prepare_hybrid_bundle import archive_bundle
    root = Path(__file__).resolve().parents[1]
    identity = uuid4().hex
    source = root / 'artifacts/hybrid/test_bundle' / identity
    destination = root / 'docs/research_artifacts/test_bundle' / identity
    source.mkdir(parents=True)
    (source / 'summary.json').write_text(json.dumps({'execution_status': 'executed',
                                                   'dataset_kind': 'synthetic_source_conformance'}))
    (source / 'dataset.json').write_text(json.dumps({'kind': 'synthetic_source_conformance', 'cases': []}))
    try:
        first = archive_bundle({'text': source}, destination)
        assert first == archive_bundle({'text': source}, destination)
        original = (destination / 'text/summary.json').read_bytes()
        (source / 'summary.json').write_text(json.dumps({'execution_status': 'changed'}))
        with pytest.raises(ValueError, match='Preserve'):
            archive_bundle({'text': source}, destination)
        assert (destination / 'text/summary.json').read_bytes() == original
    finally:
        for filename in ('text/summary.json', 'text/dataset.json', 'BUNDLE_MANIFEST.json'):
            (destination / filename).unlink(missing_ok=True)
        (destination / 'text').rmdir()
        destination.rmdir()


def test_bundle_dry_run_keeps_both_mapping_records_and_training_schedules():
    import hashlib
    import json
    from pathlib import Path
    from uuid import uuid4
    from scripts.prepare_hybrid_bundle import archive_bundle
    root = Path(__file__).resolve().parents[1]
    identity = uuid4().hex
    source = root / 'artifacts/hybrid/test_bundle' / identity
    source.mkdir(parents=True)
    record = {'status': 'mapped', 'pubchem_cid': 123, 'source_url': 'https://pubchem.ncbi.nlm.nih.gov/',
              'retrieved_at': '2026-10-05T00:00:00Z'}
    for filename in ('molecules_frozen.json', 'molecules.json'):
        (source / filename).write_text(json.dumps({'status': 'complete', 'schema': 'test',
                                                   'records': {'ingredient': record}}))
    schedules = {}
    for label in ('model_original', 'model_corrected'):
        directory = source / label
        directory.mkdir()
        (directory / 'summary.json').write_text(json.dumps({'status': 'complete', 'runs': []}))
        run = directory / 'pair/fingerprint/17'
        run.mkdir(parents=True)
        (run / 'explanations.json').write_text(json.dumps({'cases': [{'public_ingredient_id': 'ingredient'}]}))
        (run / 'missing_modalities.json').write_text(json.dumps({'count': 1}))
        snapshot = directory / 'source_snapshot'
        snapshot.mkdir()
        content = b'# execution source snapshot\n'
        (snapshot / 'trainer.py').write_bytes(content)
        (snapshot / 'manifest.json').write_text(json.dumps({'files': [
            {'path': 'trainer.py', 'current_matches_execution': True,
             'execution_sha256': hashlib.sha256(content).hexdigest()},
            {'path': 'changed.py', 'current_matches_execution': False,
             'execution_sha256': 'unrecoverable'}]}))
        schedules[label] = directory
    destination = root / 'docs/research_artifacts/test_bundle' / identity
    archive = archive_bundle({'molecules': source, **schedules}, destination, dry_run=True)
    assert archive['dry_run'] and not destination.exists()
    files = archive['files_sha256']
    assert 'molecules/molecules_frozen.json' in files
    assert 'molecules/molecules.json' in files
    for label in schedules:
        assert f'{label}/pair/fingerprint/17/explanations.json' in files
        assert f'{label}/pair/fingerprint/17/missing_modalities.json' in files
        assert f'{label}/source_snapshot/trainer.py' in files
        assert f'{label}/source_snapshot/manifest.json' in files
        assert f'{label}/source_snapshot/changed.py' not in files
    (schedules['model_corrected'] / 'source_snapshot/trainer.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='snapshot bytes'):
        archive_bundle({'model_corrected': schedules['model_corrected']}, destination, dry_run=True)
