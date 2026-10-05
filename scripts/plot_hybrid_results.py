"""Create paper-ready figures from already executed hybrid case metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hybrid_benchmark import aggregate


def plot_results(directory: Path, split: str = 'test') -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (directory / 'case_results.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if split == 'all' or r['split'] == split]
    if not rows:
        raise ValueError('No executed case metrics in the selected split.')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    methods = [('resolved_only', 'Resolved identities'), ('top1_assumption', 'Top-1 assumed identities'),
               ('candidate_propagation', 'Candidate-potential records')]
    variants = sorted({r['input_kind'] for r in rows}, key=lambda s: (s != 'clean', s))
    buckets = {(method, variant): aggregate([r for r in rows if r['method'] == method and r['input_kind'] == variant])
               for method, _ in methods for variant in variants}
    colors = ('#246c63', '#ae7734', '#496caa')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for method_index, (method, label) in enumerate(methods):
        x = [index + (method_index - 1) * .24 for index in range(len(variants))]
        for axis, metric in zip(axes, ('high_recall', 'precision')):
            axis.bar(x, [buckets[(method, v)][metric] or 0 for v in variants], .23,
                     color=colors[method_index], label=label)
            axis.set_xticks(range(len(variants)), [v.replace('_', '\n') for v in variants])
            axis.set_ylim(0, 1.08)
            axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Fraction of clean-source reference findings')
    axes[0].set_title('High-severity source-record recall')
    axes[1].set_title('Source-record precision')
    axes[1].legend(loc='lower left', fontsize=8)
    fig.suptitle(f'{split.title()} groups: potential findings retain identity uncertainty', fontsize=12)
    fig.tight_layout()
    outputs = []
    for suffix in ('png', 'svg'):
        path = directory / f'finding_recovery.{suffix}'
        fig.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(fig)
    propagated = [r for r in rows if r['method'] == 'candidate_propagation']
    sizes = sorted({r['size'] for r in propagated})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for variant in variants:
        points = [(size, aggregate([r for r in propagated if r['size'] == size and r['input_kind'] == variant]))
                  for size in sizes if any(r['size'] == size and r['input_kind'] == variant for r in propagated)]
        axes[0].plot([s for s, _ in points], [a['conditional_findings_mean'] for _, a in points],
                     marker='o', label=variant.replace('_', ' '))
        axes[1].plot([s for s, _ in points], [a['conditional_extra_findings_mean'] for _, a in points],
                     marker='o', label=variant.replace('_', ' '))
    axes[0].set_title('Candidate-potential records beyond resolved findings')
    axes[1].set_title('Conditional records absent from reference prescription')
    for axis in axes:
        axis.set_xlabel('Reference medication entries')
        axis.set_ylabel('Mean unique source records per prescription')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Conditional alert burden; these records do not confirm medicine identities', fontsize=12)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        path = directory / f'conditional_burden.{suffix}'
        fig.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    points = [(size, aggregate([r for r in propagated if r['size'] == size])) for size in sizes]
    axis.plot(sizes, [a['latency_median_ms'] for _, a in points], marker='o', label='Candidate-response cache cleared')
    axis.plot(sizes, [a['warm_latency_median_ms'] for _, a in points], marker='o', label='Repeated identical candidate request')
    axis.set(xlabel='Reference medication entries', ylabel='Median screening latency (ms)',
             title='Pairwise candidate screening; source and OS caches may be warm')
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        path = directory / f'candidate_latency.{suffix}'
        fig.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_model_results(directory: Path) -> list[Path]:
    """Plot fixed-split seed variability, not a clinical confidence interval."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete':
        raise ValueError('Model figures require a completed training summary.')
    models = ('fingerprint', 'graphsage', 'fusion', 'role_fusion')
    labels = ('Molecular', 'GraphSAGE', 'Fusion', 'Fusion + roles')
    conditions = (('pair', 'test', 'Held-out pairs'),
                  ('cold', 'test_one_unseen', 'One unseen drug'),
                  ('cold', 'test_both_unseen', 'Both drugs unseen'))
    index = {(row['protocol'], row['model']): row for row in summary['aggregate']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey='row')
    for column, (protocol, condition, title) in enumerate(conditions):
        for row_number, (metric, ylabel) in enumerate((('macro_f1', 'Macro F1'),
                                                      ('high_pr_auc', 'High-source-severity AUPRC'))):
            values = [index[(protocol, model)]['test_conditions'][condition][metric] for model in models]
            means = [value['mean'] if value is not None else float('nan') for value in values]
            errors = [value['sd'] if value is not None else 0 for value in values]
            axis = axes[row_number, column]
            axis.bar(range(len(models)), means, yerr=errors, capsize=4,
                     color=['#246c63', '#496caa', '#ad7734', '#825b8f'])
            axis.set_xticks(range(len(models)), labels, rotation=18)
            axis.set_ylim(0, 1.02)
            axis.grid(axis='y', alpha=.2)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_number == 0:
                axis.set_title(title)
    fig.suptitle('Conditional source severity: mean and SD across three training seeds', fontsize=12)
    fig.tight_layout()
    outputs = []
    for suffix in ('png', 'svg'):
        path = directory / f'classification_comparison.{suffix}'
        fig.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for axis, (protocol, condition, title) in zip(axes, conditions):
        for offset, stage, label, color in ((-.17, 'before_calibration', 'Before temperature scaling', '#ad7734'),
                                            (.17, 'after_calibration', 'After temperature scaling', '#246c63')):
            points = []
            for model in models:
                values = [row['evaluation'][condition][stage]['ece'] for row in summary['runs']
                          if row['protocol'] == protocol and row['model'] == model]
                points.append(statistics.mean(values))
            axis.bar([i + offset for i in range(len(models))], points, .32, color=color, label=label)
        axis.set_xticks(range(len(models)), labels, rotation=18)
        axis.set_title(title)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Expected calibration error')
    axes[2].legend(fontsize=8)
    fig.suptitle('Source-label confidence calibration; this is not patient-risk calibration', fontsize=12)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        path = directory / f'calibration_comparison.{suffix}'
        fig.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'artifacts/hybrid/text')
    parser.add_argument('--model-input', type=Path, help='Plot a completed model training summary instead of recognition metrics.')
    parser.add_argument('--split', choices=('all', 'development', 'calibration', 'test'), default='test')
    args = parser.parse_args()
    outputs = plot_model_results(args.model_input) if args.model_input else plot_results(args.input, args.split)
    print(json.dumps({'figures': [str(path) for path in outputs]}, indent=2))


if __name__ == '__main__':
    main()
