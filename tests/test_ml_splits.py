from __future__ import annotations

from copy import deepcopy
import pytest

from adr_system.ml.dataset import make_splits, audit_splits, audit_predictive_features
from adr_system.ml.calibration import softmax, fit_temperature, choose_threshold, wilson_upper


def examples():
    return [{'ingredient_ids': [f'd{i}', f'd{j}'], 'source_record_id': f'd{i}|d{j}',
             'severity': ('high', 'moderate', 'low')[(i + j) % 3]}
            for i in range(40) for j in range(i + 1, 40)]


def test_disjoint_message_supervision_and_cold_drug_roles():
    pairs = examples()
    splits = make_splits(pairs)
    audit = audit_splits(pairs, splits)
    assert audit['passed']
    assert splits == make_splits(pairs)
    assert set(splits['pair']['message']).isdisjoint(splits['pair']['supervision'])
    train = set(splits['cold_drugs']['train'])
    assert all(set(pairs[i]['ingredient_ids']) <= train for i in splits['cold']['message'])
    assert splits['cold']['test_both_unseen']


def test_reverse_pair_and_split_leakage_are_rejected():
    pairs = examples()
    splits = make_splits(pairs)
    duplicate = deepcopy(pairs[0])
    duplicate['ingredient_ids'].reverse()
    with pytest.raises(ValueError, match='reversed'):
        audit_splits(pairs + [duplicate], splits)
    leaked = deepcopy(splits)
    leaked['pair']['message'].append(leaked['pair']['test'][0])
    with pytest.raises(ValueError, match='leakage'):
        audit_splits(pairs, leaked)


def test_exact_molecule_aliases_share_cold_partition():
    pairs = examples()
    keys = {f'd{i}': f'chemical-{i}' for i in range(40)}
    keys['d1'] = keys['d0']
    splits = make_splits(pairs, chemical_keys=keys)
    assert any({'d0', 'd1'} <= set(ids) for ids in splits['cold_drugs'].values())
    assert audit_splits(pairs, splits)['passed']


def test_injected_source_label_or_prose_is_rejected_as_node_feature():
    clean = {'nodes': [{'ingredient_id': 'd1', 'name': 'drug', 'canonical_smiles': 'CC',
                        'inchikey': 'example', 'fingerprint_bits': [1, 2], 'roles': [1]}],
             'role_feature_names': ['biology:documented_role_present']}
    assert audit_predictive_features(clean)['passed']
    leaked = deepcopy(clean)
    leaked['nodes'][0]['severity'] = 'high'
    with pytest.raises(ValueError, match='leakage'):
        audit_predictive_features(leaked)
    leaked = deepcopy(clean)
    leaked['nodes'][0]['interaction_text'] = 'Major interaction with the target drug.'
    with pytest.raises(ValueError, match='leakage'):
        audit_predictive_features(leaked)


def test_cold_role_violation_is_rejected():
    pairs = examples()
    splits = make_splits(pairs)
    wrong = deepcopy(splits)
    message = wrong['cold']['message'][0]
    heldout = wrong['cold']['test_both_unseen'][0]
    wrong['cold']['message'][0], wrong['cold']['test_both_unseen'][0] = heldout, message
    with pytest.raises(ValueError, match='drug leakage'):
        audit_splits(pairs, wrong)


def test_temperature_improves_overconfident_source_label_log_loss():
    logits = [[12, 0, 0]] * 30 + [[0, 12, 0]] * 30
    labels = [0] * 20 + [1] * 10 + [1] * 20 + [2] * 10
    assert fit_temperature(logits, labels) > 1
    assert sum(softmax([[1000, 999, 998]])[0]) == pytest.approx(1)


def test_wilson_policy_abstains_when_sample_is_too_small_or_errors_high():
    assert choose_threshold([[.99, .005, .005]] * 99, [0] * 99)['threshold'] is None
    assert choose_threshold([[.99, .005, .005]] * 200, [1] * 200)['threshold'] is None
    policy = choose_threshold([[.99, .005, .005]] * 200, [0] * 200)
    assert policy['threshold'] == .99
    assert policy['wilson_95_upper_error'] <= .1
    assert wilson_upper(0, 100) > 0


def test_threshold_handles_confidence_ties_as_one_group():
    policy = choose_threshold([[.8, .1, .1]] * 120, [0] * 110 + [1] * 10)
    assert policy['threshold'] is None  # cannot select only the correct tied examples
