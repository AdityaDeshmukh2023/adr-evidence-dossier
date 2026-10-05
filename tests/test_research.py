import copy
import pytest
from adr_system.research import synthetic_dataset, validate_dataset, evaluate_case, edit_distance


def test_dataset_is_repeatable_and_groups_do_not_leak():
    first, second = synthetic_dataset(6), synthetic_dataset(6)
    assert first == second
    validate_dataset(first)
    broken = copy.deepcopy(first)
    broken['cases'][1]['split'] = 'test'
    with pytest.raises(ValueError, match='leak'):
        validate_dataset(broken)


def test_oracle_and_review_budget_are_consistent():
    case = synthetic_dataset(2)['cases'][2]
    rows, diagnostic = evaluate_case(case)
    oracle = [r for r in rows if r['method'] == 'gold_upper_bound']
    assert all(r['fn'] == 0 and r['fp'] == 0 for r in oracle)
    for method in ('prescription_order', 'uncertainty', 'impact'):
        records = [r for r in rows if r['method'] == method]
        assert [r['reviews'] for r in records] == [0, 1, 2, 3, 4]
        assert records[-1]['fn'] == 0 and records[-1]['fp'] == 0
    assert diagnostic['entries'] == 4


def test_reference_metrics_do_not_treat_missing_edges_as_safe():
    assert edit_distance('aspirin', 'asplrin') == 1
    assert edit_distance(['a', 'b'], ['a']) == 1
    data = synthetic_dataset(1)
    data['kind'] = 'expert_reviewed'
    with pytest.raises(ValueError, match='Permission|permission'):
        validate_dataset(data)
