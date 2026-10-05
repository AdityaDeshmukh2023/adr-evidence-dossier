"""Exact source-conformance checks. A failing case produces a nonzero exit code."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from adr_system.engine import analyze_medications
from adr_system.provenance import manifest


def evaluate(fixture: dict) -> dict:
    rows = []
    for case in fixture['cases']:
        start = perf_counter()
        result = analyze_medications(case['medications'], case.get('foods', ''))
        actual = sorted([a.source_record_id, a.interaction_type, a.severity] for a in result.alerts)
        passed = (actual == sorted(case['expected']) and result.completeness == case['completeness']
                  and all(a.evidence and a.data_version and a.confidence is None for a in result.alerts))
        rows.append({'id': case['id'], 'passed': passed, 'expected': case['expected'],
                     'actual': actual, 'completeness': result.completeness,
                     'latency_ms': (perf_counter() - start) * 1000})
    return {'snapshot': fixture['version'], 'scope': fixture['purpose'],
            'passed': sum(r['passed'] for r in rows), 'total': len(rows), 'cases': rows, 'manifest': manifest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'data/evaluation_v2.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/evaluation')
    args = parser.parse_args()
    result = evaluate(json.loads(args.fixtures.read_text(encoding='utf-8')))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('snapshot', 'passed', 'total')}, indent=2))
    for case in result['cases']:
        if not case['passed']:
            print(json.dumps(case))
    return int(result['passed'] != result['total'])


if __name__ == '__main__':
    raise SystemExit(main())
