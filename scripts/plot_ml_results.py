"""Render source-label comparison and calibration figures from executed artifacts."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=ROOT / 'artifacts/ml/training')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ml/figures')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = read_json(args.training / 'summary.json')
    args.output.mkdir(parents=True, exist_ok=True)
    models = ('fingerprint', 'graphsage', 'fusion', 'role_fusion')
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    colors = ['#427D90', '#769D57', '#D08A47', '#8A6A9C']
    for axis, protocol, condition, title in zip(axes, ('pair', 'cold', 'cold'),
            ('test', 'test_one_unseen', 'test_both_unseen'),
            ('Held-out pairs', 'One unseen drug', 'Both unseen drugs')):
        rows = [next(row for row in summary['aggregate'] if row['protocol'] == protocol and row['model'] == name)
                for name in models]
        values = [row['test_conditions'][condition]['macro_f1']['mean'] for row in rows]
        errors = [row['test_conditions'][condition]['macro_f1']['sd'] for row in rows]
        axis.bar(range(4), values, yerr=errors, color=colors, capsize=3)
        axis.set_xticks(range(4), ['FP', 'SAGE', 'Fusion', '+Roles'])
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_ylabel('Macro-F1 (mean ± seed SD)')
        axis.grid(axis='y', alpha=.2)
    figure.suptitle('Conditional source-severity prediction; three fixed seeds')
    figure.savefig(args.output / 'model_comparison.png', dpi=180)
    figure.savefig(args.output / 'model_comparison.svg')
    plt.close(figure)
    if summary['selected_path']:
        selected = read_json(Path(summary['selected_path']) / 'results.json')
        figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
        axis.plot([0, 1], [0, 1], linestyle='--', color='#999999', label='Ideal')
        for condition, color in (('before_calibration', '#D08A47'), ('after_calibration', '#427D90')):
            points = selected['evaluation']['test'][condition]['reliability']
            axis.plot([p['confidence'] for p in points], [p['accuracy'] for p in points], 'o-', color=color,
                      label=condition.replace('_', ' '))
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted confidence', ylabel='Source-label accuracy',
                 title='Held-out pair calibration')
        axis.legend()
        figure.savefig(args.output / 'reliability.png', dpi=180)
        figure.savefig(args.output / 'reliability.svg')
        plt.close(figure)
    print(json.dumps({'status': 'executed', 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
