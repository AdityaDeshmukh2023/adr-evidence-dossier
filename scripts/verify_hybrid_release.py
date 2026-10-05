"""Verify a local deployment, source authority, replay, and CPU request performance."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def process_memory() -> dict:
    """Measure native process memory, including Torch; no optional package needed."""
    if os.name == 'nt':
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError('Unable to measure process working set.')
        return {'rss_mib': counters.WorkingSetSize / 2**20,
                'peak_working_set_mib': counters.PeakWorkingSetSize / 2**20,
                'scope': 'Windows process working set; includes native ML allocations.'}
    import resource
    # Linux reports KiB, macOS reports bytes. This is a peak, not current RSS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peak_working_set_mib': peak / (2**20 if sys.platform == 'darwin' else 1024),
            'scope': 'OS-reported peak resident memory of this process.'}


def source_signature(result) -> list:
    return sorted((a.source_record_id, a.severity, tuple(e.evidence_id for e in a.evidence))
                  for a in result.alerts)


def verify(model_dir: Path, repeats: int) -> dict:
    from adr_system.engine import analyze_medications
    from adr_system.evidence_graph import build_evidence_graph
    from adr_system.explanations import explanation_payload
    from adr_system.ml.inference import bundle_manifest, predict_pairs, _load_bundle
    from adr_system.report import render_html_report, render_json_report, replay_report
    from adr_system.uncertainty import candidate_cache_info

    checks = {}
    text = 'warfarin\naspirin\nacetaminophen'
    # First model request before source/model warmup. Process startup is excluded.
    before = perf_counter()
    enabled = analyze_medications(text, enable_research_model=True, model_dir=str(model_dir))
    first_request_ms = (perf_counter() - before) * 1000
    baseline = analyze_medications(text)
    checks['source_records_preserved'] = source_signature(enabled) == source_signature(baseline)
    unknown = {a.source_record_id for a in baseline.alerts if a.severity == 'unknown'}
    checks['unknown_only_predictions'] = bool(unknown) and {
        p['source_record_id'] for p in enabled.model_predictions} == unknown
    checks['model_outputs_excluded_from_groq'] = explanation_payload(enabled) == explanation_payload(baseline)
    checks['typed_graph_valid'] = build_evidence_graph(enabled).validate() is None

    export = render_json_report(enabled)
    replay = replay_report(export, model_dir=str(model_dir))
    checks['json_replay_exact'] = (source_signature(replay) == source_signature(enabled)
                                 and replay.model_predictions == enabled.model_predictions
                                 and replay.candidate_assessments == enabled.candidate_assessments
                                 and replay.mechanism_hypotheses == enabled.mechanism_hypotheses)
    html = render_html_report(enabled, 'Local deployment verification.')
    checks['html_hybrid_sections'] = all(label in html for label in (
        'Candidate-dependent assessments', 'Mechanism-supported possibilities', 'Experimental model predictions'))

    unavailable = analyze_medications(text, enable_research_model=True,
                                     model_dir=str(ROOT / 'artifacts/__verification_absent_model__'))
    checks['absent_model_source_fallback'] = (
        source_signature(unavailable) == source_signature(baseline)
        and unavailable.model_predictions
        and all(not p['accepted'] and p['abstention_reason'] == 'model_artifacts_absent'
                for p in unavailable.model_predictions))
    uncertain = analyze_medications('Warfarln\nAspirin', enable_research_model=True, model_dir=str(model_dir))
    checks['candidate_findings_remain_conditional_on_identity'] = (
        bool(uncertain.candidate_assessments) and not uncertain.alerts and not uncertain.model_predictions)
    pathway = analyze_medications('clarithromycin\nmidazolam', "St. John's wort")
    checks['sourced_mechanism_paths_present'] = bool(pathway.mechanism_hypotheses)

    status = bundle_manifest(model_dir)
    no_record_case = None
    if status['status'] == 'available' and status['runtime_available']:
        pair = next(iter(enabled.model_predictions))['ingredient_ids']
        ab, ba = predict_pairs([pair, list(reversed(pair))], model_dir)
        checks['symmetric_model_probabilities'] = all(
            abs(ab['probabilities'][k] - ba['probabilities'][k]) <= 1e-7 for k in ab['probabilities'])
        checks['model_scores_present'] = bool(ab['probabilities'])
        from adr_system.knowledge import pair_alert
        from adr_system.terminology import catalog
        features = json.loads((model_dir / 'features.json').read_text(encoding='utf-8'))
        names = catalog()[1]
        for first, second in combinations([n['ingredient_id'] for n in features['nodes']], 2):
            if pair_alert(first, second) is None:
                no_record_case = [names[first], names[second]]
                break
        if no_record_case:
            missing = analyze_medications('\n'.join(no_record_case), enable_research_model=True,
                                          model_dir=str(model_dir))
            checks['no_record_never_predicts'] = (not missing.alerts and not missing.model_predictions
                                                and any(p['state'] == 'no_record' for p in missing.pair_assessments))

    cache_before = candidate_cache_info()._asdict()
    timings = []
    checks['repeat_outputs_stable'] = True
    for _ in range(repeats):
        before = perf_counter()
        repeated = analyze_medications(text, enable_research_model=True, model_dir=str(model_dir))
        timings.append((perf_counter() - before) * 1000)
        checks['repeat_outputs_stable'] &= repeated.model_predictions == enabled.model_predictions
    memory = process_memory()
    sorted_timings = sorted(timings)
    return {
        'status': 'passed' if all(checks.values()) else 'failed',
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checks, 'checks_passed': sum(bool(v) for v in checks.values()),
        'checks_total': len(checks), 'model': status, 'manifest': enabled.manifest,
        'predictions': enabled.model_predictions, 'no_source_record_demo': no_record_case,
        'performance': {'first_request_ms': first_request_ms, 'warm_repeats': repeats,
                        'warm_median_ms': statistics.median(timings),
                        'warm_p95_ms': sorted_timings[min(repeats - 1, int(repeats * .95))],
                        'warm_samples_ms': timings,
                        'scope': 'Fresh-process first source/model request, then same three-entry request with warm caches; excludes Python launch, Streamlit rendering, OCR and live services.'},
        'memory': memory, 'logical_cpus': os.cpu_count(),
        'candidate_cache': {'before': cache_before, 'after': candidate_cache_info()._asdict()},
        'model_cache': _load_bundle.cache_info()._asdict(),
        'scope': 'Local software and artifact correctness; not clinical validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'artifacts/ml/deployment')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/hybrid-release/model_integration.json')
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be at least two')
    result = verify(args.model_dir.resolve(), args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'checks_passed', 'checks_total', 'performance', 'memory')}, indent=2))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
