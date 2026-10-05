"""Offline source-conformance benchmark for candidate-propagating screening.

Reference identities are used only for dataset construction and scoring. No
reference text or identity is passed into normalization or screening.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import random
import statistics
import sys
import tracemalloc
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adr_system.data import load_ddinter_lookup
from adr_system.knowledge import pair_alert, pair_state
from adr_system.normalization import normalise_medication
from adr_system.provenance import manifest
from adr_system.research import gold_ids, source_reference
from adr_system.terminology import exact

SIZES = (2, 5, 10, 20, 30)
METHODS = ('resolved_only', 'top1_assumption', 'candidate_propagation')
SEVERITIES = {'high', 'moderate', 'low', 'unknown'}


def split_for_group(index: int, strata: int = 5) -> str:
    """Stratify group splits within each prescription-size bucket."""
    fold = (index // strata) % 10
    return 'development' if fold < 6 else 'calibration' if fold < 8 else 'test'


def _corrupt(name: str, kind: str, rng: random.Random) -> str:
    if kind == 'clean' or len(name) < 4:
        return name
    pos = rng.randrange(1, len(name) - 1)
    if kind == 'single_character_error':
        replacement = 'l' if name[pos] != 'l' else 'i'
        return name[:pos] + replacement + name[pos + 1:]
    operation = rng.choice(('substitute', 'delete', 'transpose', 'ocr_confusion'))
    if operation == 'delete':
        return name[:pos] + name[pos + 1:]
    if operation == 'transpose':
        return name[:pos] + name[pos + 1] + name[pos] + name[pos + 2:]
    if operation == 'ocr_confusion':
        value = name.translate(str.maketrans({'o': '0', 'i': '1', 's': '5'}))
        if value != name:
            return value
    return name[:pos] + 'x' + name[pos + 1:]


def synthetic_dataset(groups: int = 500, seed: int = 1729) -> dict:
    if groups < 1:
        raise ValueError('groups must be positive')
    lookup, _ = load_ddinter_lookup()
    if not lookup:
        raise ValueError('Generate the full DDInter source snapshot first.')
    rng = random.Random(seed)
    names = sorted(n for n in lookup['drug_index']
                   if 4 <= len(n) <= 30 and not any(c in n for c in ',;\n')
                   and len(exact(n)) == 1 and len(exact(n)[0].ingredient_ids) == 1)
    eligible = set(names)
    buckets = {s: [] for s in sorted(SEVERITIES)}
    for row in lookup['interactions'].values():
        if row['drug_a_name'].casefold() in eligible and row['drug_b_name'].casefold() in eligible:
            buckets[row['severity']].append(row)
    present = [s for s in ('high', 'moderate', 'low', 'unknown') if buckets[s]]
    cases = []
    seen_clean = set()
    for group in range(groups):
        size = SIZES[group % len(SIZES)]
        severity = present[group % len(present)]
        for _ in range(1000):
            anchor = rng.choice(buckets[severity])
            drugs = [anchor['drug_a_name'].casefold(), anchor['drug_b_name'].casefold()]
            drugs.extend(rng.sample([n for n in names if n not in drugs], size - 2))
            rng.shuffle(drugs)
            if tuple(drugs) not in seen_clean:
                seen_clean.add(tuple(drugs))
                break
        else:
            raise ValueError('Cannot generate enough distinct prescription groups.')
        reference_entries = [{'gold_names': [name]} for name in drugs]
        findings = source_reference(reference_entries)
        for variant in ('clean', 'single_character_error', 'mixed_corruption'):
            entries = []
            for index, name in enumerate(drugs):
                value = _corrupt(name, variant, rng) if index % 2 == 0 else name
                if variant == 'mixed_corruption' and group % 13 == 0 and index == size - 1:
                    value = 'zxqvvv'  # deliberately includes candidate-generation failures
                entries.append({'text': value, 'gold_names': [name]})
            cases.append({'id': f'HYB-{group:04d}-{variant}', 'group_id': f'H{group:04d}',
                          'split': split_for_group(group), 'input_kind': variant,
                          'size': size, 'entries': entries, 'gold_findings': findings})
    return {'version': 'hybrid-synthetic-1.0', 'kind': 'synthetic_source_conformance',
            'seed': seed, 'base_groups': groups, 'sizes': list(SIZES),
            'label_provenance': 'Clean ingredient identities and independent local source-table lookup.',
            'scope': 'Synthetic text; entry boundaries supplied; no clinical outcome labels.',
            'split_policy': '60/20/20 development/calibration/test by base group within size bucket.',
            'cases': cases}


def validate_dataset(dataset: dict) -> None:
    if dataset.get('kind') != 'synthetic_source_conformance':
        raise ValueError('This runner accepts the declared synthetic source-conformance dataset only.')
    cases = dataset.get('cases', [])
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Case IDs must be unique and nonempty.')
    groups, fingerprints = {}, {}
    for case in cases:
        split = case['split']
        if split not in ('development', 'calibration', 'test'):
            raise ValueError('Unknown dataset split.')
        if groups.setdefault(case['group_id'], split) != split:
            raise ValueError('Related prescription variants leak across splits.')
        fingerprint = tuple(e['text'].strip().casefold() for e in case['entries'])
        if fingerprints.setdefault(fingerprint, split) != split:
            raise ValueError('Identical observations leak across splits.')
        if len(case['entries']) not in SIZES:
            raise ValueError('Prescription size is outside the specified benchmark strata.')
        for entry in case['entries']:
            if not entry['text'].strip() or not gold_ids(entry):
                raise ValueError('Each synthetic entry needs observation text and a valid reference identity.')


def finding_key(record: dict) -> tuple[tuple[str, ...], str]:
    return tuple(sorted(record['ingredient_ids'])), record['severity']


def _findings_for_ids(ids) -> set:
    findings = set()
    for a, b in itertools.combinations(sorted(set(ids)), 2):
        alert = pair_alert(a, b)
        if alert:
            findings.add((tuple(alert.ingredient_ids), alert.severity))
    return findings


def _options(medication):
    if medication.ingredient_ids:
        return (medication.ingredient_ids,)
    return tuple(c.ingredient_ids for c in medication.candidates)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_findings(predicted: set, gold: set) -> dict:
    high = {item for item in gold if item[1] == 'high'}
    return {'tp': len(predicted & gold), 'fp': len(predicted - gold), 'fn': len(gold - predicted),
            'high_tp': len(predicted & high), 'high_total': len(high),
            'finding_total': len(gold), 'predicted_total': len(predicted)}


def evaluate_medications(case: dict, medications: list, *, gold_by_mention: dict | None = None,
                         gold_entry_count: int | None = None, profile_memory: bool = False) -> list[dict]:
    """Score extracted medications without making reference-assisted corrections.

    gold_by_mention is a scoring-only alignment. Image callers construct it from
    rendered bounding boxes AFTER OCR, never from labels during inference.
    """
    from adr_system.uncertainty import (candidate_assessments, candidate_cache_info,
                                        clear_candidate_cache)
    if gold_by_mention is None:
        gold_by_mention = {f'm{i + 1}': gold_ids(e) for i, e in enumerate(case['entries'])}
    gold_entry_count = gold_entry_count if gold_entry_count is not None else len(case['entries'])
    gold = {finding_key(f) for f in case['gold_findings']}
    base = {'case_id': case['id'], 'group_id': case['group_id'], 'split': case['split'],
            'input_kind': case['input_kind'], 'size': case.get('size', gold_entry_count),
            'entries': gold_entry_count, 'detected_entries': len(medications),
            'candidate_hits': sum(any(ids == gold_by_mention.get(m.mention_id) for ids in _options(m))
                                  for m in medications),
            'identity_correct': sum(bool(m.ingredient_ids) and m.ingredient_ids == gold_by_mention.get(m.mention_id)
                                    for m in medications),
            'candidate_count': sum(len(_options(m)) for m in medications),
            'no_candidate_entries': sum(not _options(m) for m in medications)}
    rows = []
    for method in METHODS[:2]:
        start = perf_counter()
        selected = [m.ingredient_ids if m.ingredient_ids else
                    (m.candidates[0].ingredient_ids if method == 'top1_assumption' and m.candidates else ())
                    for m in medications]
        predictions = _findings_for_ids(i for ids in selected for i in ids)
        rows.append({**base, 'method': method, **score_findings(predictions, gold),
                     'latency_ms': (perf_counter() - start) * 1000,
                     'result_semantics': 'assumed identities' if method == 'top1_assumption' else 'resolved identities',
                     'conditional_findings': 0, 'stable_predicates': 0,
                     'incorrect_stable_predicates': 0, 'unscorable_stable_predicates': 0})
    # Cold here means the candidate-response cache, NOT source-file or OS caches.
    clear_candidate_cache()
    if profile_memory:
        tracemalloc.start()
    start = perf_counter()
    assessments = candidate_assessments(medications)
    cold_ms = (perf_counter() - start) * 1000
    peak = None
    if profile_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    cache_cold = candidate_cache_info()._asdict()
    start = perf_counter()
    warm = candidate_assessments(medications)
    warm_ms = (perf_counter() - start) * 1000
    if assessments != warm:
        raise AssertionError('Candidate-response cache changed the screening result.')
    cache_warm = candidate_cache_info()._asdict()
    confirmed = _findings_for_ids(i for m in medications for i in m.ingredient_ids)
    predictions = confirmed | {finding_key(record) for assessment in assessments
                               for outcome in assessment['outcomes'] for record in outcome['records']}
    stable_count = incorrect = unscorable = 0
    for assessment in assessments:
        ids_a = gold_by_mention.get(assessment['mention_ids'][0])
        ids_b = gold_by_mention.get(assessment['mention_ids'][1])
        reference_states = {pair_state(a, b) for a in (ids_a or ()) for b in (ids_b or ())}
        for state in set(assessment['stable_states']) & SEVERITIES:
            stable_count += 1
            if not ids_a or not ids_b:
                unscorable += 1
            elif state not in reference_states:
                incorrect += 1
    rows.append({**base, 'method': METHODS[2], **score_findings(predictions, gold),
                 'latency_ms': cold_ms, 'warm_latency_ms': warm_ms,
                 'python_peak_bytes': peak, 'response_cache_cold': cache_cold,
                 'response_cache_warm': cache_warm,
                 'result_semantics': 'potential source records conditional on retained candidates',
                 'confirmed_tp': len(confirmed & gold), 'confirmed_fp': len(confirmed - gold),
                 'conditional_findings': len(predictions - confirmed),
                 'conditional_extra_findings': len(predictions - confirmed - gold),
                 'stable_predicates': stable_count, 'incorrect_stable_predicates': incorrect,
                 'unscorable_stable_predicates': unscorable,
                 'unusable_entry_pairs': sum(a['status'] == 'unusable_identity' for a in assessments),
                 'candidate_combinations': sum(len(a['outcomes']) for a in assessments)})
    return rows


def evaluate_case(case: dict, *, profile_memory: bool = False) -> list[dict]:
    start = perf_counter()
    observed = [replace(normalise_medication(e['text']), mention_id=f'm{i + 1}')
                for i, e in enumerate(case['entries'])]
    normalization_ms = (perf_counter() - start) * 1000
    rows = evaluate_medications(case, observed, profile_memory=profile_memory)
    for row in rows:
        row['normalization_ms'] = normalization_ms
        row['end_to_end_ms'] = normalization_ms + row['latency_ms']
    return rows


def _quantile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(rows: list[dict]) -> dict:
    def total(key):
        return sum(r.get(key, 0) or 0 for r in rows)
    tp, fp, fn = total('tp'), total('fp'), total('fn')
    return {'cases': len(rows), 'groups': len({r['group_id'] for r in rows}),
            'precision': _ratio(tp, tp + fp), 'recall': _ratio(tp, tp + fn),
            'high_recall': _ratio(total('high_tp'), total('high_total')),
            'f1': _ratio(2 * tp, 2 * tp + fp + fn),
            'candidate_coverage': _ratio(total('candidate_hits'), total('entries')),
            'exact_identity_accuracy': _ratio(total('identity_correct'), total('entries')),
            'candidate_set_size_mean': _ratio(total('candidate_count'), total('detected_entries')),
            'conditional_findings_mean': _ratio(total('conditional_findings'), len(rows)),
            'conditional_extra_findings_mean': _ratio(total('conditional_extra_findings'), len(rows)),
            'incorrect_stable_rate': _ratio(total('incorrect_stable_predicates'),
                                           total('stable_predicates') - total('unscorable_stable_predicates')),
            'stable_predicates': total('stable_predicates'),
            'unscorable_stable_predicates': total('unscorable_stable_predicates'),
            'latency_median_ms': statistics.median(r['latency_ms'] for r in rows),
            'latency_p95_ms': _quantile([r['latency_ms'] for r in rows], .95),
            'normalization_median_ms': statistics.median(r.get('normalization_ms', 0) for r in rows),
            'warm_latency_median_ms': _quantile([r['warm_latency_ms'] for r in rows if 'warm_latency_ms' in r], .5),
            'python_peak_bytes_max': max((r['python_peak_bytes'] for r in rows
                                          if r.get('python_peak_bytes') is not None), default=None)}


def paired_bootstrap(rows: list[dict], repeats: int = 1000, seed: int = 7) -> dict:
    """Pair-resample independent TEST prescription groups; variants stay together."""
    groups = {}
    for row in rows:
        if row['split'] == 'test' and row['input_kind'] != 'clean':
            group = groups.setdefault(row['group_id'], {})
            count = group.setdefault(row['method'], [0, 0])
            count[0] += row['high_tp']
            count[1] += row['high_total']
    paired = [g for g in groups.values() if METHODS[0] in g and METHODS[2] in g]
    if len(paired) < 2 or not sum(g[METHODS[0]][1] for g in paired):
        return {'difference': None, 'ci95': None, 'reason': 'Insufficient test groups or source high-severity labels.'}
    def delta(sample):
        base = sum(g[METHODS[0]][0] for g in sample) / sum(g[METHODS[0]][1] for g in sample)
        proposed = sum(g[METHODS[2]][0] for g in sample) / sum(g[METHODS[2]][1] for g in sample)
        return proposed - base
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        sample = rng.choices(paired, k=len(paired))
        if sum(g[METHODS[0]][1] for g in sample):
            values.append(delta(sample))
    return {'comparison': 'candidate potential high-source recall minus resolved-only recall',
            'difference': delta(paired), 'ci95': [_quantile(values, .025), _quantile(values, .975)],
            'test_groups': len(paired), 'paired_group_bootstrap_repeats': repeats, 'seed': seed,
            'interpretation': 'Potential findings include conditional identities; see precision and burden alongside recall.'}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def write_outputs(dataset: dict, rows: list[dict], output: Path, *, run_manifest: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dataset, indent=2, sort_keys=True)
    (output / 'dataset.json').write_text(serialized, encoding='utf-8')
    (output / 'case_results.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows), encoding='utf-8')
    write_csv(output / 'case_metrics.csv', rows)
    buckets = {}
    for row in rows:
        key = (row['split'], row['input_kind'], row['method'], row['size'])
        buckets.setdefault(key, []).append(row)
    comparison = [{'split': split, 'input_kind': kind, 'method': method, 'size': size, **aggregate(part)}
                  for (split, kind, method, size), part in sorted(buckets.items())]
    write_csv(output / 'comparison.csv', comparison)
    summary = {'execution_status': 'executed', 'dataset_kind': dataset['kind'],
               'dataset_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
               'cases_evaluated': len({r['case_id'] for r in rows}),
               'independent_groups': len({r['group_id'] for r in rows}),
               'primary_test_comparison': paired_bootstrap(rows), 'comparison': comparison,
               'hardware': {'platform': platform.platform(), 'python': platform.python_version()},
               'manifest': run_manifest if run_manifest is not None else manifest(),
               'limitations': ['Synthetic source conformance is not clinical validation.',
                              'Potential candidate findings are not confirmed medicine identities.',
                              'Top-1 results assume identities; no labels are used to select candidates.',
                              'No source record does not establish safety.',
                              'Cold/warm timing clears only the candidate-response cache; loaded source/OS caches stay warm.',
                              'tracemalloc samples Python allocation peaks, not complete process RSS.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'RESULTS.md').write_text(
        '# Hybrid screening benchmark\n\nExecuted synthetic source-conformance run. '
        'Candidate-potential recovery, confirmed finding recovery and stable-predicate correctness have different meanings. '
        'Read precision and conditional burden alongside recall.\n\n'
        + json.dumps(summary['primary_test_comparison'], indent=2) + '\n\n'
        'See comparison.csv and case_metrics.csv for size, split and noise strata.\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--split', choices=('all', 'development', 'calibration', 'test'), default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/hybrid/text')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--memory-samples', type=int, default=1,
                        help='Cases per size/noise stratum profiled with Python tracemalloc (slower).')
    args = parser.parse_args()
    run_manifest = manifest()  # Hash source/code before inference, not after a long running job.
    dataset = json.loads(args.dataset.read_text(encoding='utf-8')) if args.dataset else synthetic_dataset(args.groups, args.seed)
    validate_dataset(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.generate_only:
        (args.output / 'dataset.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')
        print(json.dumps({'execution_status': 'dataset_generated_only', 'groups': dataset['base_groups'],
                          'cases': len(dataset['cases'])}))
        return 0
    rows, profiled = [], {}
    cases = [c for c in dataset['cases'] if args.split == 'all' or c['split'] == args.split]
    for number, case in enumerate(cases, 1):
        key = (case['size'], case['input_kind'])
        profile = profiled.get(key, 0) < args.memory_samples
        profiled[key] = profiled.get(key, 0) + int(profile)
        rows.extend(evaluate_case(case, profile_memory=profile))
        if number % 50 == 0:
            print(json.dumps({'completed_cases': number, 'total_cases': len(cases)}), flush=True)
    if not rows:
        raise ValueError('Selected split contains no cases.')
    summary = write_outputs(dataset, rows, args.output, run_manifest=run_manifest)
    print(json.dumps({k: summary[k] for k in ('cases_evaluated', 'independent_groups', 'primary_test_comparison')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
