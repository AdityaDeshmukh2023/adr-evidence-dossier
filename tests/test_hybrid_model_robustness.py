import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hybrid_model_robustness import (CLASSES, VARIANTS, audit_holdout, evaluate_case,
                                            make_dataset, object_hash, summarize)


def documented_source(*ids):
    return 'unknown'


def fixture_source():
    source = {'nodes': [{'ingredient_id': 'a', 'name': 'alpha'},
                        {'ingredient_id': 'b', 'name': 'beta'},
                        {'ingredient_id': 'c', 'name': 'gamma'}],
              'pairs': [{'ingredient_ids': ['a', 'b'], 'severity': 'high', 'source_record_id': 'a|b'},
                        {'ingredient_ids': ['a', 'c'], 'severity': 'moderate', 'source_record_id': 'a|c'},
                        {'ingredient_ids': ['b', 'c'], 'severity': 'low', 'source_record_id': 'b|c'}]}
    splits = {'pair': {'test': [0], 'message': [1], 'supervision': [2],
                       'validation': [], 'calibration': []}}
    return source, splits


def fixture_case():
    return {'id': 'fixture', 'group_id': 'fixture-group', 'split': 'test', 'input_kind': 'clean',
            'entries': [{'text': 'alpla'}, {'text': 'beta'}],
            'reference': {'ingredient_ids': ['a', 'b'], 'source_record_id': 'a|b', 'source_severity': 'high'}}


def medication(*ids):
    return SimpleNamespace(ingredient_ids=(), candidates=tuple(
        SimpleNamespace(ingredient_ids=(ident,), score=1 - index / 10) for index, ident in enumerate(ids)))


def fake_predictor(classes=None, accepted=False, calls=None):
    classes = classes or {}
    def predict(pairs, model_dir):
        if calls is not None:
            calls.extend(copy.deepcopy(pairs))
        output = []
        for pair in pairs:
            key = '|'.join(sorted(pair))
            label = classes.get(key, 'high')
            output.append({'ingredient_ids': pair, 'source_record_id': key, 'predicted_severity': label,
                           'probabilities': {name: .8 if name == label else .1 for name in CLASSES},
                           'accepted': accepted,
                           'abstention_reason': None if accepted else 'prediction_below_calibrated_threshold'})
        return output
    return predict


def test_generation_uses_only_test_pairs_and_keeps_related_variants_together():
    source, splits = fixture_source()
    dataset = make_dataset(source, splits, count=1, seed=17)
    assert dataset == make_dataset(source, splits, count=1, seed=17)
    assert len(dataset['cases']) == 3
    assert {case['input_kind'] for case in dataset['cases']} == set(VARIANTS)
    assert {case['reference']['source_record_id'] for case in dataset['cases']} == {'a|b'}
    assert {case['group_id'] for case in dataset['cases']} == {'M0000'}
    clean = dataset['cases'][0]
    assert clean['entries'] == [{'text': 'alpha'}, {'text': 'beta'}]
    for case in dataset['cases'][1:]:
        assert all(entry['text'] != original['text'] for entry, original in zip(case['entries'], clean['entries']))
    assert dataset['holdout_audit']['true_query_training_overlap'] == 0


def test_reversed_duplicate_and_message_supervision_leakage_are_rejected():
    source, splits = fixture_source()
    splits['pair']['message'].append(0)
    with pytest.raises(ValueError, match='leakage'):
        audit_holdout(source, splits)
    source, splits = fixture_source()
    source['pairs'][1] = {'ingredient_ids': ['b', 'a'], 'severity': 'high', 'source_record_id': 'a|b'}
    with pytest.raises(ValueError, match='Duplicate or reversed'):
        audit_holdout(source, splits)
    source, splits = fixture_source()
    source['pairs'][0]['source_record_id'] = 'b|a'
    with pytest.raises(ValueError, match='Noncanonical'):
        audit_holdout(source, splits)


def test_model_supervision_provenance_is_audited():
    source, splits = fixture_source()
    scope = {'purpose': 'source_label_benchmark', 'message_pair_keys': ['a|c'],
             'supervision_pair_keys_sha256': object_hash(['b|c']), 'split_sha256': object_hash(splits)}
    assert audit_holdout(source, splits, scope)['passed']
    scope['supervision_pair_keys_sha256'] = object_hash(['a|b'])
    with pytest.raises(ValueError, match='provenance'):
        audit_holdout(source, splits, scope)


def test_gold_never_enters_normalizer_and_is_not_a_candidate_oracle():
    observed, calls = [], []
    def normalize(text):
        assert isinstance(text, str)
        observed.append(text)
        return medication('c') if text == 'alpla' else medication('b')
    original = fixture_case()
    untouched = copy.deepcopy(original)
    row = evaluate_case(original, model_ids={'a', 'b', 'c'}, normalizer=normalize,
                        predictor=fake_predictor(calls=calls), source_checker=documented_source)
    assert observed == ['alpla', 'beta']
    assert calls == [['b', 'c']]  # The missing true identity 'a' is never inserted.
    assert row['candidate_identity_hits'] == row['top1_identity_hits'] == 1
    assert row['candidate_set']['status'] == 'stable'
    assert not row['candidate_true_pair_covered']
    assert original == untouched
    changed = copy.deepcopy(original)
    changed['reference'].update(ingredient_ids=['c', 'b'], source_record_id='b|c', source_severity='low')
    calls.clear()
    other = evaluate_case(changed, model_ids={'a', 'b', 'c'}, normalizer=normalize,
                          predictor=fake_predictor(calls=calls), source_checker=documented_source)
    assert calls == [['b', 'c']]
    assert other['top1'] == row['top1']
    assert other['candidate_set'] == row['candidate_set']


def test_stability_requires_every_candidate_class_and_preserves_calibrated_abstention():
    normalize = lambda text: medication('a', 'c') if text == 'alpla' else medication('b')
    stable = evaluate_case(fixture_case(), model_ids={'a', 'b', 'c'}, normalizer=normalize,
                           predictor=fake_predictor(), source_checker=documented_source)
    assert stable['candidate_set']['status'] == 'stable'
    assert stable['candidate_set']['model_available_pairs'] == 2
    assert stable['top1']['available'] and not stable['top1']['accepted']
    assert not stable['candidate_set']['accepted']
    disagree = evaluate_case(fixture_case(), model_ids={'a', 'b', 'c'}, normalizer=normalize,
                             predictor=fake_predictor({'b|c': 'low'}, accepted=True), source_checker=documented_source)
    assert disagree['candidate_set']['status'] == 'conditional'
    assert disagree['candidate_set']['predicted_source_severity'] is None
    assert not disagree['candidate_set']['accepted']


def test_missing_molecules_are_counted_and_cannot_be_dropped_to_claim_stability():
    calls = []
    normalize = lambda text: medication('c', 'a') if text == 'alpla' else medication('b')
    row = evaluate_case(fixture_case(), model_ids={'a', 'b'}, normalizer=normalize,
                        predictor=fake_predictor(calls=calls), source_checker=documented_source)
    assert calls == [['a', 'b']]
    assert not row['top1']['available']  # No availability-assisted top-1 reranking.
    assert row['top1']['missing_molecule_ids'] == ['c']
    assert row['candidate_set']['status'] == 'conditional'
    assert row['candidate_set']['missing_molecule_pairs'] == 1
    assert row['candidate_set']['missing_molecule_ids'] == ['c']


def test_unresolved_entries_stay_unusable_and_in_all_coverage_denominators():
    row = evaluate_case(fixture_case(), model_ids={'a', 'b'}, predictor=fake_predictor(),
                        normalizer=lambda text: medication() if text == 'alpla' else medication('b'),
                        source_checker=documented_source)
    assert row['no_candidate_entries'] == 1
    assert row['candidate_set']['status'] == 'unusable'
    def metrics(probabilities, labels):
        return {'macro_f1': None, 'per_class': {}, 'high_pr_auc': None}
    summary = summarize([row], metrics)[0]
    assert summary['candidate_true_identity_coverage'] == {'numerator': 1, 'denominator': 2, 'rate': .5}
    assert summary['top1']['model_availability']['denominator'] == 1
    assert summary['top1']['raw_model_argmax']['eligible_count'] == 0
    assert summary['top1']['calibrated_decisions']['accepted_coverage']['rate'] == 0
    assert summary['top1']['calibrated_decisions']['accepted_error']['rate'] is None
    assert summary['candidate_set']['raw_stable_model_class']['high_pr_auc'] is None


def test_candidate_retention_is_bounded_without_using_labels_or_lexical_probabilities():
    calls = []
    normalize = lambda text: medication('c', 'd', 'e', 'f', 'g', 'a') if text == 'alpla' else medication('b')
    row = evaluate_case(fixture_case(), model_ids=set('abcdefg'), normalizer=normalize,
                        predictor=fake_predictor(calls=calls), source_checker=documented_source)
    assert row['retained_candidates'][0] == [['c'], ['d'], ['e'], ['f'], ['g']]
    assert row['candidate_set']['combinations'] == 5
    assert row['candidate_identity_hits'] == 1
    assert row['top1']['probabilities'] == {'high': .8, 'moderate': .1, 'low': .1}
    assert ['a', 'b'] not in calls


def test_generation_reports_missing_reference_molecules():
    source, splits = fixture_source()
    source['nodes'] = source['nodes'][1:]
    with pytest.raises(ValueError, match='only 0 TEST pairs'):
        make_dataset(source, splits, count=1)


def test_raw_model_metrics_are_separate_from_calibrated_coverage_and_error():
    normalize = lambda text: medication('a') if text == 'alpla' else medication('b')
    row = evaluate_case(fixture_case(), model_ids={'a', 'b'}, normalizer=normalize,
                        predictor=fake_predictor({'a|b': 'low'}), source_checker=documented_source)
    def metrics(probabilities, labels):
        assert labels == [0] if labels else labels == []
        return {'macro_f1': 0 if labels else None, 'per_class': {}, 'high_pr_auc': .25 if labels else None}
    report = summarize([row], metrics)[0]
    assert report['top1']['model_availability']['rate'] == 1
    assert report['top1']['raw_model_argmax']['eligible_count'] == 1
    assert report['top1']['raw_model_argmax']['high_pr_auc'] == .25
    assert report['top1']['calibrated_decisions']['accepted_coverage']['rate'] == 0
    assert report['top1']['calibrated_decisions']['accepted_error']['rate'] is None
    assert report['candidate_set']['raw_stable_model_class']['high_pr_auc'] is None
    row['top1']['accepted'] = True
    report = summarize([row], metrics)[0]
    assert report['top1']['calibrated_decisions']['accepted_coverage']['rate'] == 1
    assert report['top1']['calibrated_decisions']['accepted_error'] == {'numerator': 1, 'denominator': 1, 'rate': 1}


def test_unrecorded_alternative_never_reaches_predictor_or_becomes_a_stable_class():
    calls = []
    normalize = lambda text: medication('c', 'a') if text == 'alpla' else medication('b')
    def source(first, second):
        return 'high' if set((first, second)) == {'a', 'b'} else 'no_record'
    row = evaluate_case(fixture_case(), model_ids={'a', 'b', 'c'}, normalizer=normalize,
                        predictor=fake_predictor(calls=calls), source_checker=source)
    assert calls == [['a', 'b']]
    assert row['top1']['source_record_present'] is False
    assert row['top1']['abstention_reason'] == 'no_source_record'
    assert not row['top1']['available']
    assert row['candidate_set']['status'] == 'conditional'
    assert row['candidate_set']['no_source_record_pairs'] == 1
    assert row['candidate_set']['eligible_documented_mapped_pairs'] == 1
    assert row['candidate_set']['predicted_source_severity'] is None
    assert not row['candidate_set']['accepted']


def test_help_and_pure_scoring_work_without_torch_or_sklearn():
    root = Path(__file__).resolve().parents[1]
    command = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'sklearn'):
        raise ImportError('Optional ML packages disabled for this test')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['scripts/hybrid_model_robustness.py', '--help']
runpy.run_path('scripts/hybrid_model_robustness.py', run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-c', command], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--count' in result.stdout
    assert '--model-dir' in result.stdout
