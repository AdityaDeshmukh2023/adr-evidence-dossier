"""Offline, deterministic review simulation and source-conformance metrics.

Gold labels are read only by the simulated reviewer and metric calculation.
They are never passed into candidate generation or prioritization.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import replace
from time import perf_counter

from .data import ROOT, load_ddinter_lookup, load_interactions
from .engine import analyze_structured, apply_review
from .normalization import normalise_medication
from .terminology import exact

METHODS = ('legacy_baseline', 'corrected_baseline', 'prescription_order', 'uncertainty', 'impact', 'impact_without_severity', 'gold_upper_bound')
BUDGETS = (0., .25, .5, .75, 1.)


def gold_ids(entry: dict) -> tuple[str, ...]:
    ids = []
    for name in entry['gold_names']:
        options = exact(name)
        if len(options) != 1:
            raise ValueError(f'Gold identity is absent or ambiguous in terminology: {name}')
        ids.extend(options[0].ingredient_ids)
    return tuple(sorted(set(ids)))


def source_reference(entries: list[dict]) -> list[dict]:
    """Independent table lookup for synthetic labels, not a clinical reference."""
    ids = sorted({i for entry in entries for i in gold_ids(entry)})
    lookup, _ = load_ddinter_lookup()
    curated = {}
    for key, record in load_interactions()[0]['interactions'].items():
        options = [exact(n) for n in key.split('|')]
        if all(len(o) == 1 for o in options):
            pair = tuple(sorted(o[0].ingredient_ids[0] for o in options))
            curated[pair] = record
    reference = []
    for a, b in itertools.combinations(ids, 2):
        row = lookup['interactions'].get(f'{a}|{b}') if lookup else None
        if row is None:
            row = curated.get((a, b))
        if row:
            reference.append({'ingredient_ids': [a, b], 'severity': row['severity']})
    return reference


def synthetic_dataset(groups: int = 40, seed: int = 17) -> dict:
    lookup, _ = load_ddinter_lookup()
    if not lookup:
        raise ValueError('Synthetic benchmark requires the full source snapshot.')
    rng = random.Random(seed)
    buckets = {s: [] for s in ('high', 'moderate', 'low', 'unknown')}
    for row in lookup['interactions'].values():
        if all(len(row[k]) <= 30 and ',' not in row[k] for k in ('drug_a_name', 'drug_b_name')):
            buckets[row['severity']].append(row)
    names = sorted(lookup['drug_index'])
    cases = []
    for group in range(groups):
        severity = ('high', 'moderate', 'low', 'unknown')[group % 4]
        row = rng.choice(buckets[severity])
        drugs = [row['drug_a_name'].lower(), row['drug_b_name'].lower()]
        while len(drugs) < 4:
            name = rng.choice(names)
            if name not in drugs and len(name) <= 24 and ',' not in name:
                drugs.append(name)
        rng.shuffle(drugs)
        split = 'development' if group % 5 == 0 else 'test'
        for variant in ('clean', 'single_character_error', 'mixed_corruption'):
            entries = []
            for i, drug in enumerate(drugs):
                observed = drug
                if variant != 'clean' and i % 2 == 0 and len(drug) > 3:
                    pos = max(1, len(drug) // 2)
                    observed = drug[:pos] + ('l' if drug[pos] != 'l' else 'i') + drug[pos + 1:]
                if variant == 'mixed_corruption' and i == 1:
                    observed = drug.translate(str.maketrans({'o': '0', 'i': '1'}))
                if variant == 'mixed_corruption' and group % 7 == 0 and i == 3:
                    observed = 'ZXQVVV'  # required manual correction, gold need not be in top five
                entries.append({'text': observed, 'gold_names': [drug]})
            cases.append({'id': f'SYN-{group:03d}-{variant}', 'group_id': f'G{group:03d}', 'split': split,
                          'input_kind': variant, 'origin': 'synthetic_text', 'entries': entries,
                          'gold_findings': source_reference(entries)})
    return {'version': 'synthetic-review-1.0', 'kind': 'synthetic', 'seed': seed,
            'label_provenance': 'Local source tables; no independent clinical review.',
            'scope': 'Annotated medication-entry boundaries; text corruption is not real image OCR.', 'cases': cases}


def validate_dataset(dataset: dict):
    if dataset.get('kind') not in ('synthetic', 'expert_reviewed'):
        raise ValueError('Declare dataset kind as synthetic or expert_reviewed.')
    if dataset['kind'] == 'expert_reviewed' and not dataset.get('permission_confirmed'):
        raise ValueError('Document permission before evaluating prescription examples.')
    cases = dataset.get('cases', [])
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty and IDs unique.')
    splits, inputs = {}, {}
    for case in cases:
        if case['split'] not in ('development', 'test'):
            raise ValueError('Use development or test split.')
        prior = splits.setdefault(case['group_id'], case['split'])
        if prior != case['split']:
            raise ValueError('Related cases leak across development/test splits.')
        if not 1 <= len(case['entries']) <= 30:
            raise ValueError('Each case requires 1–30 annotated entries.')
        if dataset['kind'] == 'expert_reviewed' and case.get('annotation_status') != 'reviewed':
            raise ValueError('Expert evaluation requires reviewed annotations for every case.')
        if dataset['kind'] == 'expert_reviewed' and not case.get('reviewer_ids'):
            raise ValueError('Record pseudonymous reviewer IDs for reviewed cases.')
        fingerprint = tuple(e['text'].casefold().strip() for e in case['entries'])
        if inputs.setdefault(fingerprint, case['split']) != case['split']:
            raise ValueError('Identical input leaks across dataset splits.')
        for entry in case['entries']:
            if not entry.get('text', '').strip():
                raise ValueError('Annotation entry text cannot be blank.')
            if not entry['gold_names'] and entry.get('is_medication', True):
                raise ValueError('Medication entries need gold identities; mark non-medication text explicitly.')
            gold_ids(entry)
        if 'gold_findings' not in case:
            raise ValueError('Explicit gold findings are required; they are not inferred for expert cases.')
        for finding in case['gold_findings']:
            if len(set(finding['ingredient_ids'])) != 2 or finding['severity'] not in ('high', 'moderate', 'low', 'unknown'):
                raise ValueError('Gold finding requires two different ingredients and a valid severity.')
            if dataset['kind'] == 'expert_reviewed' and not finding.get('source_reference'):
                raise ValueError('Expert findings require a source reference.')


def finding_set(result):
    return {(tuple(sorted(a.ingredient_ids)), a.severity) for a in result.alerts if a.interaction_type == 'drug-drug'}


def legacy_baseline(case):
    from legacy.frozen_v1.adr_system import data as old_data
    from legacy.frozen_v1.adr_system.engine import analyze_medications as old_analyze
    from .models import Medication, AnalysisResult
    old_data.DATA_DIR = ROOT / 'data'
    old = old_analyze('\n'.join(e['text'] for e in case['entries']))
    medications = []
    for i, med in enumerate(old.medications):
        match = exact(med.canonical_name) if med.canonical_name else ()
        ids = match[0].ingredient_ids if len(match) == 1 else ()
        medications.append(Medication(med.name, med.canonical_name, normalized=bool(ids),
            mention_id=f'm{i + 1}', ingredient_ids=ids, status='exact' if ids else 'unresolved'))
    for alert in old.alerts:
        alert.ingredient_ids = sorted({ident for name in alert.entities for c in exact(name) for ident in c.ingredient_ids})
    return AnalysisResult(old.run_id, medications, [], alerts=old.alerts)


def metrics(result, gold: set) -> dict:
    predicted = finding_set(result)
    high_gold = {x for x in gold if x[1] == 'high'}
    return {'tp': len(predicted & gold), 'fp': len(predicted - gold), 'fn': len(gold - predicted),
            'high_tp': len(predicted & high_gold), 'high_total': len(high_gold),
            'unresolved': sum(not m.normalized and m.status != 'excluded' for m in result.medications),
            'predictions': [{'ingredient_ids': list(ids), 'severity': severity} for ids, severity in sorted(predicted)]}


def _next_mention(result, method: str):
    queue = result.review_queue
    if not queue:
        return None
    order = {m.mention_id: i for i, m in enumerate(result.medications)}
    if method == 'prescription_order':
        return min(queue, key=lambda q: (not q.required, order[q.mention_id])).mention_id
    if method == 'uncertainty':
        return min(queue, key=lambda q: (not q.required, -q.ambiguity, order[q.mention_id])).mention_id
    if method == 'impact_without_severity':
        return min(queue, key=lambda q: (not q.required, -(q.high_changes + q.other_changes),
                                        -q.coverage_changes, -q.ambiguity, order[q.mention_id])).mention_id
    return queue[0].mention_id


def evaluate_case(case: dict, methods=METHODS, budgets=BUDGETS) -> tuple[list[dict], dict]:
    start = perf_counter()
    observed = [replace(normalise_medication(e['text']), mention_id=f'm{i + 1}') for i, e in enumerate(case['entries'])]
    extraction_ms = (perf_counter() - start) * 1000
    gold_per_mention = {f'm{i + 1}': gold_ids(e) for i, e in enumerate(case['entries'])}
    gold = {(tuple(sorted(f['ingredient_ids'])), f['severity']) for f in case['gold_findings']}
    diagnostic = {'case_id': case['id'], 'group_id': case['group_id'], 'input_kind': case['input_kind'],
                  'entries': len(observed), 'exact_identity_correct': sum(m.ingredient_ids == gold_per_mention[m.mention_id] for m in observed),
                  'candidate_hits': sum(any(c.ingredient_ids == gold_per_mention[m.mention_id] for c in m.candidates) for m in observed),
                  'no_candidates': sum(not m.candidates for m in observed), 'extraction_ms': extraction_ms}
    rows = []
    for method in methods:
        start = perf_counter()
        result = legacy_baseline(case) if method == 'legacy_baseline' else analyze_structured(observed)
        states = [(result, (perf_counter() - start) * 1000, [])]
        actions = []
        if method not in ('corrected_baseline', 'legacy_baseline'):
            for _ in observed:
                ident = _next_mention(result, method)
                if ident is None:
                    break
                ids = gold_per_mention[ident]  # oracle is consulted only AFTER selection
                result = apply_review(result, ident, ids, action='resolve' if ids else 'exclude_non_medication')
                actions = actions + [ident]
                states.append((result, (perf_counter() - start) * 1000, actions))
        for budget in budgets:
            requested = math.ceil(len(observed) * budget)
            k = len(states) - 1 if method == 'gold_upper_bound' else min(requested, len(states) - 1)
            state, latency, actions_at_budget = states[k]
            row = {'case_id': case['id'], 'group_id': case['group_id'], 'split': case['split'],
                   'input_kind': case['input_kind'], 'method': method, 'budget': budget,
                   'reviews': k, 'latency_ms': latency + extraction_ms, 'actions': actions_at_budget,
                   **metrics(state, gold)}
            row['identity_correct'] = sum(m.ingredient_ids == gold_per_mention[m.mention_id] for m in state.medications)
            row['entries'] = len(observed)
            rows.append(row)
    return rows, diagnostic


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]
