"""Archive executed hybrid experiments with original manifests and SHA-256 hashes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FILES = ('summary.json', 'dataset.json', 'comparison.csv', 'case_metrics.csv',
                   'case_results.jsonl', 'image_metrics.csv', 'synthetic_ocr_outputs.jsonl',
                   'RESULTS.md', 'finding_recovery.png', 'finding_recovery.svg',
                   'conditional_burden.png', 'conditional_burden.svg',
                   'candidate_latency.png', 'candidate_latency.svg')
MODEL_FILES = ('summary.json', 'metrics.json', 'dataset_manifest.json', 'training_manifest.json',
               'evaluation.json', 'calibration.json', 'model.pt', 'checkpoint.pt', 'model_card.md',
               'splits.json', 'predictions.csv', 'test_predictions.csv', 'RESULTS.md',
               'manifest.json', 'weights.pt', 'features.json', 'graph.json', 'config.json',
               'runs.json', 'results.json', 'training_scope.json', 'coverage_report.json',
               'explanation_outputs.json', 'explanation_metrics.json', 'explanations.json',
               'split_audit.json', 'model_card.json', 'coverage.json', 'selection.json',
               'missing_modalities.json', 'case_results.jsonl', 'case_metrics.csv', 'comparison.csv',
               'explanation_outputs.jsonl', 'explanation_cases.json', 'fidelity.json',
               'bookkeeping_corrections.json', 'calibration_raw_before_bookkeeping.json',
               'manifest_raw_before_bookkeeping.json', 'test_predictions.json', 'feature_audit.json')
PER_RUN_FILES = {'results.json', 'calibration.json', 'training_scope.json', 'config.json',
                 'model_card.json', 'manifest.json', 'missing_modalities.json',
                 'explanations.json', 'explanation_outputs.json', 'explanation_metrics.json',
                 'explanation_outputs.jsonl', 'explanation_cases.json', 'fidelity.json',
                 'calibration_raw_before_bookkeeping.json', 'manifest_raw_before_bookkeeping.json',
                 'bookkeeping_corrections.json', 'test_predictions.json'}
MOLECULE_FILES = ('molecules_frozen.json', 'molecules.json', 'full_catalog_mapping_summary.json',
                  'aggregation_benchmark_final.json')
MODEL_FIGURE_FILES = ('model_comparison.png', 'model_comparison.svg', 'reliability.png', 'reliability.svg')
RELEASE_FILES = ('graphsage_optimization.json', 'compute_sample.json', 'compute_sample_minibatch.json', 'startup.json',
                 'preliminary_model_integration.json', 'preliminary_model_ui.json',
                 'core_integration.json', 'model_integration.json', 'model_ui.json', 'verification.json',
                 'model_integration_idle.json', 'source_checks.json',
                 'tests.json', 'source_fixtures.json', 'dependency_checks.json')


def archive_bundle(sources: dict[str, Path], destination: Path, *, include_images: bool = False,
                   dry_run: bool = False) -> dict:
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / 'docs/research_artifacts').resolve()):
        raise ValueError('Store reviewable paper bundles in docs/research_artifacts.')
    pending, derived_contents, experiments, frozen_dataset, deployment_manifest = [], [], {}, None, None
    for name, directory in sources.items():
        if not name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Experiment names must be simple directory labels.')
        directory = directory.resolve()
        if not directory.is_relative_to((ROOT / 'artifacts').resolve()):
            raise ValueError('Read experiment artifacts from the workspace artifacts directory.')
        if name == 'molecules':
            mapping_metadata = {}
            for filename in MOLECULE_FILES:
                path = directory / filename
                if not path.is_file():
                    continue
                pending.append((path, destination / name / filename))
                if filename in ('molecules_frozen.json', 'molecules.json'):
                    mapping = json.loads(path.read_text(encoding='utf-8'))
                    mapping_metadata[filename] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'status': mapping.get('status'), 'schema': mapping.get('schema'),
                        'catalog_size': mapping.get('catalog_size'),
                        'record_statuses': dict(Counter(row.get('status') for row in mapping.get('records', {}).values()))}
            experiments[name] = {'included': bool(mapping_metadata), 'execution_status': 'snapshots_preserved',
                'snapshots': mapping_metadata,
                'scope': 'Full public PubChem mapping records and separately scoped computation evidence; frozen training mapping is distinct from completed catalog mapping.'}
            continue
        if name == 'model_figures':
            included = []
            for filename in MODEL_FIGURE_FILES:
                path = directory / filename
                if path.is_file():
                    pending.append((path, destination / name / filename))
                    included.append(filename)
            experiments[name] = {'included': bool(included), 'execution_status': 'figures_preserved',
                                  'files': included, 'scope': 'Plots of the corrected source-label experiment.'}
            continue
        if name == 'release':
            included = []
            for filename in RELEASE_FILES:
                path = directory / filename
                if path.is_file():
                    pending.append((path, destination / name / filename))
                    included.append(filename)
            experiments[name] = {'included': bool(included), 'execution_status': 'verification_records_preserved',
                                  'files': included, 'scope': 'Each verification artifact retains its original measured scope.'}
            continue
        if name == 'baseline':
            included = []
            for filename in ('manifest.json', 'evaluation/results.json'):
                path = directory / filename
                if path.is_file():
                    pending.append((path, destination / name / filename))
                    included.append(filename)
            experiments[name] = {'included': bool(included), 'execution_status': 'baseline_records_preserved',
                                  'files': included, 'scope': 'Original code/data hashes and automatic source fixtures.'}
            continue
        is_model = name in ('model', 'model_original', 'model_corrected', 'model_diagnostic',
                            'deployment', 'explanations', 'ml_dataset', 'model_recognition')
        summary_path = directory / 'summary.json'
        if is_model and not summary_path.exists():
            summary_path = next((directory / p for p in ('manifest.json', 'runs.json', 'results.json', 'dataset_manifest.json', 'dataset.json')
                                 if (directory / p).is_file()), summary_path)
        if name == 'model_diagnostic' and not summary_path.exists():
            summary_path = next(iter(sorted(directory.glob('*/results.json'))), summary_path)
        if not summary_path.exists():
            experiments[name] = {'execution_status': 'no_summary_found', 'included': False}
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if name == 'deployment' and summary_path.name == 'manifest.json':
            deployment_manifest = summary
            for filename, expected_hash in summary.get('files', {}).items():
                path = (directory / filename).resolve()
                if not path.is_relative_to(directory) or not path.is_file():
                    raise ValueError('Deployment manifest references an absent or external file.')
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError('Deployment checkpoint files disagree with their original manifest.')
        if name == 'ml_dataset' and summary_path.name == 'dataset.json':
            frozen_dataset = summary
            frozen = {key: summary.get(key) for key in ('schema', 'task', 'classes', 'fingerprint',
                      'content_sha256', 'source_csv_sha256', 'molecule_cache_sha256', 'biology_manifest',
                      'class_counts', 'exclusions', 'role_feature_names', 'scope')}
            frozen.update({'original_dataset_file_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                           'node_count': len(summary.get('nodes', [])),
                           'pair_count': len(summary.get('pairs', [])),
                           'derivation': 'Metadata copied from the frozen dataset; full features/labels retained in original artifacts.'})
            derived_contents.append((destination / name / 'dataset_manifest.json',
                                     json.dumps(frozen, indent=2, sort_keys=True).encode('utf-8')))
        dataset_path = directory / 'dataset.json'
        if dataset_path.exists() and not is_model:
            dataset = json.loads(dataset_path.read_text(encoding='utf-8'))
            if dataset.get('kind') != 'synthetic_source_conformance':
                raise ValueError('Only declared synthetic source-conformance datasets may be bundled.')
        allowed = MODEL_FILES if is_model else BENCHMARK_FILES
        if name == 'model_recognition':
            allowed += ('dataset.json',)
        for filename in allowed:
            path = directory / filename
            if path.is_file():
                pending.append((path, destination / name / filename))
        if name in ('model', 'model_original', 'model_corrected', 'model_diagnostic', 'explanations'):
            for path in sorted(directory.rglob('*')):
                if path.is_file() and ((path.parent != directory and path.name in PER_RUN_FILES)
                                       or path.suffix in ('.png', '.svg')):
                    resolved = path.resolve()
                    if not resolved.is_relative_to(directory):
                        raise ValueError('Model run artifact resolves outside its experiment directory.')
                    pending.append((path, destination / name / path.relative_to(directory)))
            snapshot = directory / 'source_snapshot'
            snapshot_manifest = snapshot / 'manifest.json'
            if snapshot_manifest.is_file():
                snapshot_data = json.loads(snapshot_manifest.read_text(encoding='utf-8'))
                for row in snapshot_data.get('files', []):
                    if not row.get('current_matches_execution'):
                        continue
                    path = (snapshot / row['path']).resolve()
                    if not path.is_relative_to(snapshot.resolve()) or not path.is_file():
                        raise ValueError('Execution source snapshot references an absent or external file.')
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row['execution_sha256']:
                        raise ValueError('Execution source snapshot bytes differ from captured execution hashes.')
                    pending.append((path, destination / name / 'source_snapshot' / path.relative_to(snapshot)))
                # The manifest is already included by the per-run allowlist. Its false
                # rows document files that could not honestly be recovered after execution.
        if include_images and dataset_path.exists() and not is_model and dataset.get('modality') == 'actual_image_ocr':
            for case in dataset['cases']:
                path = Path(case['image']).resolve()
                if not path.is_relative_to(directory) or path.suffix.lower() != '.png':
                    raise ValueError('Synthetic image path lies outside its experiment directory.')
                if hashlib.sha256(path.read_bytes()).hexdigest() != case['image_sha256']:
                    raise ValueError('Image bytes differ from the experiment dataset manifest.')
                pending.append((path, destination / name / 'images' / path.name))
        experiments[name] = {'included': True, 'execution_status': summary.get('execution_status', summary.get('status', 'unspecified')),
                             'original_summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                             'scope': summary.get('scope', summary.get('dataset_kind', 'See original manifest.'))}
    if not pending and not derived_contents:
        raise ValueError('No executed experiment files were available to bundle.')
    integrity_checks = {}
    mapping_snapshots = experiments.get('molecules', {}).get('snapshots', {})
    if frozen_dataset and mapping_snapshots:
        matches = [name for name, metadata in mapping_snapshots.items()
                   if metadata['sha256'] == frozen_dataset.get('molecule_cache_sha256')]
        if not matches:
            raise ValueError('The archived molecule caches do not match the frozen training dataset.')
        integrity_checks['training_molecule_cache'] = {'status': 'matching', 'files': matches}
    if frozen_dataset and deployment_manifest:
        if deployment_manifest.get('dataset_sha256') != frozen_dataset.get('content_sha256'):
            raise ValueError('Selected deployment and frozen dataset versions disagree.')
        integrity_checks['selected_deployment_dataset'] = 'matching'
    contents = [(target, source.read_bytes()) for source, target in pending] + derived_contents
    for target, content in contents:
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Preserve the existing archive; choose a new output directory for changed results.')
    entries = {target.relative_to(destination).as_posix(): hashlib.sha256(content).hexdigest()
               for target, content in contents}
    archive = {'schema_version': '1.0', 'scope': 'Executed source-conformance experiments and optional research-model outputs.',
               'experiments': experiments, 'files_sha256': entries,
               'integrity_checks': integrity_checks,
               'provenance_rule': 'Original run summaries are preserved; this archive does not rerun or reinterpret them.',
               'limitations': 'No clinical outcome validation is implied by model or synthetic benchmark metrics.'}
    serialized = json.dumps(archive, indent=2, sort_keys=True).encode('utf-8')
    manifest_path = destination / 'BUNDLE_MANIFEST.json'
    if manifest_path.exists() and manifest_path.read_bytes() != serialized:
        raise ValueError('Preserve the existing bundle manifest; choose a new output directory.')
    if dry_run:
        return {**archive, 'dry_run': True, 'total_bytes': sum(len(content) for _, content in contents)}
    for target, content in contents:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest_path.write_bytes(serialized)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-directory', type=Path, default=ROOT / 'artifacts/hybrid/text')
    parser.add_argument('--ocr-directory', type=Path, default=ROOT / 'artifacts/hybrid/ocr')
    parser.add_argument('--model-directory', type=Path)
    parser.add_argument('--original-model-directory', type=Path)
    parser.add_argument('--corrected-model-directory', type=Path)
    parser.add_argument('--diagnostic-directory', type=Path)
    parser.add_argument('--model-figures-directory', type=Path, default=ROOT / 'artifacts/ml/figures-minibatch')
    parser.add_argument('--molecules-directory', type=Path, default=ROOT / 'artifacts/ml')
    parser.add_argument('--release-directory', type=Path, default=ROOT / 'artifacts/hybrid-release')
    parser.add_argument('--baseline-directory', type=Path, default=ROOT / 'artifacts/hybrid-baseline')
    parser.add_argument('--model-recognition-directory', type=Path)
    parser.add_argument('--deployment-directory', type=Path, default=ROOT / 'artifacts/ml/deployment')
    parser.add_argument('--explanation-directory', type=Path)
    parser.add_argument('--ml-dataset-directory', type=Path, default=ROOT / 'artifacts/ml/dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/research_artifacts/hybrid_checkpoint_2026-10-05')
    parser.add_argument('--include-images', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate hashes/paths and report size without writing an archive.')
    args = parser.parse_args()
    sources = {'text': args.text_directory, 'ocr': args.ocr_directory}
    if args.model_directory:
        sources['model'] = args.model_directory
    if args.original_model_directory:
        sources['model_original'] = args.original_model_directory
    if args.corrected_model_directory:
        sources['model_corrected'] = args.corrected_model_directory
    if args.diagnostic_directory:
        sources['model_diagnostic'] = args.diagnostic_directory
    if args.model_figures_directory:
        sources['model_figures'] = args.model_figures_directory
    if args.molecules_directory:
        sources['molecules'] = args.molecules_directory
    if args.release_directory:
        sources['release'] = args.release_directory
    if args.baseline_directory:
        sources['baseline'] = args.baseline_directory
    if args.model_recognition_directory:
        sources['model_recognition'] = args.model_recognition_directory
    if args.deployment_directory:
        sources['deployment'] = args.deployment_directory
    if args.explanation_directory:
        sources['explanations'] = args.explanation_directory
    if args.ml_dataset_directory:
        sources['ml_dataset'] = args.ml_dataset_directory
    archive = archive_bundle(sources, args.output, include_images=args.include_images, dry_run=args.dry_run)
    print(json.dumps({'output': str(args.output), 'files': len(archive['files_sha256']),
                      'dry_run': archive.get('dry_run', False), 'total_bytes': archive.get('total_bytes'),
                      'experiments': archive['experiments']}, indent=2))


if __name__ == '__main__':
    main()
