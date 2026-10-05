"""Archive an allowlist of synthetic experiment outputs for the writing team."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = {
    'research': ['summary.json', 'dataset.json', 'comparison.csv', 'case_metrics.csv',
                 'case_results.jsonl', 'ablation_at_half_budget.csv', 'extraction_metrics.csv',
                 'failure_categories.csv', 'identity_summary.csv', 'latency_cost.csv',
                 'severity_confusion.csv', 'RESULTS.md', 'review_budget.png', 'review_budget.svg',
                 'identity_resolution.png', 'failure_categories.png'],
    'ocr_benchmark': ['summary.json', 'image_metrics.csv', 'ocr_error_rates.png'],
    'evaluation': ['results.json'],
}
TEXT_EXTENSIONS = {'.json', '.jsonl', '.csv', '.md', '.svg'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'docs/research_artifacts/checkpoint_2026-10-04')
    parser.add_argument('--live-directory', type=Path,
                        help='Optional synthetic output directory from explanation_benchmark.py.')
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT / 'docs/research_artifacts').resolve()):
        raise ValueError('Store paper bundles within docs/research_artifacts.')
    sources = [(ROOT / 'artifacts' / folder, folder, names) for folder, names in EXPORTS.items()]
    for folder in ('research', 'ocr_benchmark'):
        summary = json.loads((ROOT / 'artifacts' / folder / 'summary.json').read_text(encoding='utf-8'))
        if summary.get('dataset_kind') != 'synthetic':
            raise ValueError('Only synthetic benchmark results may be archived by this command.')
    dataset = json.loads((ROOT / 'artifacts/research/dataset.json').read_text(encoding='utf-8'))
    if dataset.get('kind') != 'synthetic':
        raise ValueError('Refusing to archive a private/expert dataset.')
    if args.live_directory:
        live = args.live_directory.resolve()
        if not live.is_relative_to((ROOT / 'artifacts').resolve()):
            raise ValueError('Read explanation artifacts from the local artifacts directory.')
        outputs = json.loads((live / 'outputs.json').read_text(encoding='utf-8'))
        if not outputs or any(o['case_id'] not in {f'EXPL-{i}' for i in range(1, 5)} for o in outputs):
            raise ValueError('Only the four fixed synthetic explanation examples are accepted.')
        sources.append((live, 'explanations_live', ['summary.json', 'outputs.json',
                                                  'blinded_review.json', 'ratings.csv']))
    pending = []
    for directory, name, filenames in sources:
        for filename in filenames:
            path = directory / filename
            content = path.read_bytes()
            if path.suffix in TEXT_EXTENSIONS and re.search(rb'gsk_[A-Za-z0-9]{20,}', content):
                raise ValueError(f'Credential-like content detected in {name}/{filename}; no bundle written.')
            target = destination / name / filename
            if target.exists() and target.read_bytes() != content:
                raise ValueError('An archived output differs. Select a new --output directory; preserve old runs.')
            pending.append((path, target, content))
    entries = {}
    for source, target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries[target.relative_to(destination).as_posix()] = hashlib.sha256(content).hexdigest()
    archive = {'scope': 'Synthetic results and unfilled human-rating sheets only; no clinical validation.',
               'files_sha256': entries,
               'provenance': 'Each experiment summary retains its own original execution manifest.',
               'rule': 'Never reinterpret an old manifest as representing a later code revision.'}
    (destination / 'BUNDLE_MANIFEST.json').write_text(json.dumps(archive, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(destination.relative_to(ROOT)), 'files': len(entries)}, indent=2))


if __name__ == '__main__':
    main()
