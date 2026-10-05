"""Exact pair-level source screening within retained medicine candidates.

No similarity scores are interpreted as probabilities. A stable severity is a
predicate over these retained interpretations, never a confirmed medicine ID.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

from .knowledge import pair_alert
from .models import Medication


def _identities(med: Medication) -> tuple[tuple[str, ...], ...]:
    if med.status == 'excluded':
        return ()
    if med.ingredient_ids:
        return (tuple(sorted(set(med.ingredient_ids))),)
    return tuple(dict.fromkeys(tuple(sorted(set(c.ingredient_ids)))
                              for c in med.candidates[:5] if c.ingredient_ids))


@lru_cache(maxsize=8192)
def _joins(first: tuple[tuple[str, ...], ...], second: tuple[tuple[str, ...], ...]):
    outcomes = []
    for a, b in itertools.product(first, second):
        checks = []
        for x, y in sorted(set(tuple(sorted(p)) for p in itertools.product(a, b))):
            alert = pair_alert(x, y) if x != y else None
            state = 'same_ingredient' if x == y else alert.severity if alert else 'no_record'
            checks.append(((x, y), state, alert.source_record_id if alert else None))
        outcomes.append((a, b, tuple(checks)))
    return tuple(outcomes)


@lru_cache(maxsize=2048)
def _internal_joins(choices: tuple[tuple[str, ...], ...]):
    outcomes = []
    for ids in choices:
        checks = []
        for first, second in itertools.combinations(ids, 2):
            alert = pair_alert(first, second)
            checks.append(((first, second), alert.severity if alert else 'no_record',
                           alert.source_record_id if alert else None))
        outcomes.append((ids, tuple(checks)))
    return tuple(outcomes)


def candidate_assessments(medications: list[Medication]) -> list[dict]:
    assessments = []
    active = [m for m in medications if m.status != 'excluded']
    for a, b in itertools.combinations(active, 2):
        first, second = _identities(a), _identities(b)
        base = {'mention_ids': [a.mention_id, b.mention_id],
                'scope': 'retained_candidates_only', 'candidate_counts': [len(first), len(second)],
                'enumeration_complete': True}
        if not first or not second:
            assessments.append({**base, 'status': 'unusable_identity', 'outcomes': [],
                                'stable_states': [], 'possible_states': ['unassessed'],
                                'stable_record_ids': []})
            continue
        outcomes, states, records = [], [], []
        for ids_a, ids_b, checks in _joins(first, second):
            rows = [{'ingredient_ids': list(ids), 'state': state, 'source_record_id': record}
                    for ids, state, record in checks]
            findings = [{**row, 'severity': row['state']} for row in rows if row['source_record_id']]
            present = {row['state'] for row in rows}
            states.append(present)
            records.append({row['source_record_id'] for row in findings})
            outcomes.append({'candidate_a': list(ids_a), 'candidate_b': list(ids_b),
                             'states': sorted(present), 'records': findings, 'pair_checks': rows})
        stable = set.intersection(*states)
        possible = set.union(*states)
        status = ('resolved' if a.ingredient_ids and b.ingredient_ids else
                  'stable' if all(s == states[0] for s in states) else 'conditional')
        assessments.append({**base, 'status': status, 'outcomes': outcomes,
                            'stable_states': sorted(stable), 'possible_states': sorted(possible),
                            'stable_record_ids': sorted(set.intersection(*records))})
    # A combination product can contain an interaction even when it is the only
    # prescription entry. Do not lose these records for uncertain product IDs.
    for medication in active:
        choices = _identities(medication)
        if not any(len(ids) > 1 for ids in choices):
            continue
        outcomes, states, records = [], [], []
        for ids, checks in _internal_joins(choices):
            rows = [{'ingredient_ids': list(pair), 'state': state, 'source_record_id': record}
                    for pair, state, record in checks]
            findings = [{**row, 'severity': row['state']} for row in rows if row['source_record_id']]
            present = {row['state'] for row in rows} or {'no_ingredient_pair'}
            states.append(present)
            records.append({row['source_record_id'] for row in findings})
            outcomes.append({'candidate_a': list(ids), 'candidate_b': [],
                             'states': sorted(present), 'records': findings, 'pair_checks': rows})
        status = ('resolved' if medication.ingredient_ids else
                  'stable' if all(s == states[0] for s in states) else 'conditional')
        assessments.append({'mention_ids': [medication.mention_id],
                            'scope': 'within_entry_retained_candidates_only',
                            'candidate_counts': [len(choices)], 'enumeration_complete': True,
                            'status': status, 'outcomes': outcomes,
                            'stable_states': sorted(set.intersection(*states)),
                            'possible_states': sorted(set.union(*states)),
                            'stable_record_ids': sorted(set.intersection(*records))})
    return assessments


def candidate_cache_info():
    external, internal = _joins.cache_info(), _internal_joins.cache_info()
    return type(external)(external.hits + internal.hits, external.misses + internal.misses,
                          external.maxsize + internal.maxsize, external.currsize + internal.currsize)


def clear_candidate_cache():
    _joins.cache_clear()
    _internal_joins.cache_clear()
