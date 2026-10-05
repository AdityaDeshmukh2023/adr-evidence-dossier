"""Reproducible review-budget experiments; defaults to explicitly synthetic data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adr_system.provenance import manifest
from adr_system.research import synthetic_dataset, validate_dataset, evaluate_case


def aggregate(frame):
    tp, fp, fn = frame.tp.sum(), frame.fp.sum(), frame.fn.sum()
    total = frame.high_total.sum()
    return {'precision': float(tp / (tp + fp)) if tp + fp else None,
            'recall': float(tp / (tp + fn)) if tp + fn else None,
            'high_recall': float(frame.high_tp.sum() / total) if total else None,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
            'identity_accuracy': float(frame.identity_correct.sum() / frame.entries.sum()),
            'reviews_mean': float(frame.reviews.mean()), 'latency_median_ms': float(frame.latency_ms.median()),
            'latency_p95_ms': float(frame.latency_ms.quantile(.95)), 'unresolved_mean': float(frame.unresolved.mean())}


def paired_interval(frame, first='impact', second='uncertainty', budget=.5):
    subset = frame[(frame.budget == budget) & (frame.input_kind != 'clean')]
    a = subset[subset.method == first].groupby('group_id')[['high_tp', 'high_total']].sum()
    b = subset[subset.method == second].groupby('group_id')[['high_tp', 'high_total']].sum().reindex(a.index)
    if len(a) < 2 or not a.high_total.sum():
        return {'difference': None, 'ci95': None, 'reason': 'Insufficient independent groups / high-severity references.'}
    rng = np.random.default_rng(17)
    values = []
    for _ in range(1000):
        indices = rng.integers(0, len(a), len(a))
        aa, bb = a.iloc[indices], b.iloc[indices]
        if aa.high_total.sum() and bb.high_total.sum():
            values.append(aa.high_tp.sum() / aa.high_total.sum() - bb.high_tp.sum() / bb.high_total.sum())
    difference = a.high_tp.sum() / a.high_total.sum() - b.high_tp.sum() / b.high_total.sum()
    return {'difference': float(difference), 'ci95': np.quantile(values, [.025, .975]).tolist(),
            'groups': len(a), 'resampling': '1000 paired prescription-group bootstrap samples; seed 17',
            'comparison': f'{first} minus {second}; corrupted text; budget {budget}'}


def write_outputs(dataset, rows, diagnostics, output):
    frame, diag = pd.DataFrame(rows), pd.DataFrame(diagnostics)
    frame.to_json(output / 'case_results.jsonl', orient='records', lines=True)
    frame.drop(columns=['actions', 'predictions']).to_csv(output / 'case_metrics.csv', index=False)
    diag.to_csv(output / 'extraction_metrics.csv', index=False)
    records = []
    for (method, budget, kind), group in frame.groupby(['method', 'budget', 'input_kind']):
        records.append({'method': method, 'budget': budget, 'input_kind': kind, **aggregate(group)})
    table = pd.DataFrame(records)
    table.to_csv(output / 'comparison.csv', index=False)
    table[table.budget == .5].to_csv(output / 'ablation_at_half_budget.csv', index=False)
    latency = frame.groupby('method').latency_ms.agg(['median', lambda s: s.quantile(.95)])
    latency.columns = ['median_ms', 'p95_ms']
    latency['external_requests'] = 0
    latency['api_cost_usd'] = 0
    latency.to_csv(output / 'latency_cost.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9, 5))
    noisy = frame[frame.input_kind != 'clean']
    for method, group in noisy.groupby('method'):
        points = [(budget, aggregate(part)['high_recall']) for budget, part in group.groupby('budget')]
        ax.plot([p[0] for p in points], [p[1] for p in points], marker='o', label=method.replace('_', ' '))
    ax.set(xlabel='Fraction of entries reviewed (simulated)', ylabel='High-severity source-record recall', ylim=(-.03, 1.03),
           title=f'{dataset["kind"]}: source finding recovery under text corruption')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'review_budget.png', dpi=220); fig.savefig(output / 'review_budget.svg'); plt.close(fig)
    errors = pd.Series({'No candidates': int(diag.no_candidates.sum()),
                        'Gold absent from top 5': int(diag.entries.sum() - diag.candidate_hits.sum()),
                        'Not resolved correctly before review': int(diag.entries.sum() - diag.exact_identity_correct.sum())})
    errors.to_csv(output / 'failure_categories.csv', header=['count'])
    fig, ax = plt.subplots(figsize=(9, 4)); errors.plot.barh(ax=ax, color='#24756c')
    ax.set(title='Overlapping failure categories; synthetic text is not image OCR', xlabel='Medication entries')
    fig.tight_layout(); fig.savefig(output / 'failure_categories.png', dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    identity = diag.groupby('input_kind')[['entries', 'exact_identity_correct', 'candidate_hits']].sum()
    identity['exact_accuracy'] = identity.exact_identity_correct / identity.entries
    identity['candidate_recall_at_5'] = identity.candidate_hits / identity.entries
    identity.to_csv(output / 'identity_summary.csv')
    identity[['exact_accuracy', 'candidate_recall_at_5']].plot.bar(ax=ax, color=['#24756c', '#bc8734'])
    ax.set(ylim=(0, 1.05), title='Medication-entry identity resolution'); ax.tick_params(axis='x', rotation=15)
    fig.tight_layout(); fig.savefig(output / 'identity_resolution.png', dpi=220); plt.close(fig)
    # Severity confusion is over pair identities; missing references are NOT safe negatives.
    cases = {c['id']: c for c in dataset['cases']}
    confusion = []
    for row in rows:
        if row['budget'] != .5 or row['method'] not in ('uncertainty', 'impact'):
            continue
        gold = {tuple(sorted(f['ingredient_ids'])): f['severity'] for f in cases[row['case_id']]['gold_findings']}
        predicted = {tuple(sorted(f['ingredient_ids'])): f['severity'] for f in row['predictions']}
        for pair in gold.keys() | predicted.keys():
            confusion.append({'method': row['method'], 'reference': gold.get(pair, 'not_in_reference'),
                              'prediction': predicted.get(pair, 'not_detected')})
    pd.DataFrame(confusion, columns=['method', 'reference', 'prediction']).value_counts().rename('count').to_csv(output / 'severity_confusion.csv')
    summary = {'dataset_version': dataset['version'], 'dataset_kind': dataset['kind'],
               'scope': dataset.get('scope'), 'label_provenance': dataset.get('label_provenance'),
               'cases_evaluated': len(diag), 'independent_groups': int(diag.group_id.nunique()),
               'primary_comparison': paired_interval(frame), 'manifest': manifest(),
               'limitations': ['Simulated correction reveals gold labels; it does not measure clinician time.',
                              'Medication-entry boundaries are supplied; use the image benchmark for OCR.',
                              'Source-table labels do not constitute clinical validation.',
                              'No external LLM requests were made by this experiment.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'RESULTS.md').write_text('# Review simulation results\n\nDataset: '+dataset['kind']+'\n\n'+
        'These are source-conformance and simulated-review results. No claim of clinical accuracy or measured clinician time is supported.\n\n'+
        json.dumps(summary['primary_comparison'], indent=2)+'\n\nSee comparison.csv, case_results.jsonl, identity_summary.csv, severity_confusion.csv and latency_cost.csv.\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/research')
    parser.add_argument('--groups', type=int, default=40)
    parser.add_argument('--split', choices=['development', 'test'], default='test')
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding='utf-8')) if args.dataset else synthetic_dataset(args.groups)
    validate_dataset(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dataset, indent=2)
    (args.output / 'dataset.json').write_text(serialized, encoding='utf-8')
    rows, diagnostics = [], []
    for case in dataset['cases']:
        if case['split'] == args.split:
            case_rows, diagnostic = evaluate_case(case)
            rows.extend(case_rows); diagnostics.append(diagnostic)
    if not rows:
        raise ValueError('Selected split has no cases.')
    summary = write_outputs(dataset, rows, diagnostics, args.output)
    summary['dataset_sha256'] = hashlib.sha256(serialized.encode()).hexdigest()
    summary['split'] = args.split
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('dataset_kind', 'cases_evaluated', 'primary_comparison')}, indent=2))


if __name__ == '__main__':
    main()
