from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from urllib.parse import urlparse

from .data import load_food_rules
from .models import AnalysisResult, Candidate, Medication, ReviewDecision, SCHEMA_VERSION
from .provenance import manifest

DISCLAIMER = 'Educational decision-support result only. Do not start, stop, or change medication without a qualified clinician.'


def safe_url(url: str) -> str:
    return url if urlparse(url).scheme in ('http', 'https') else '#'


def export_dict(result: AnalysisResult, *, include_original: bool = False) -> dict:
    from .evidence_graph import build_evidence_graph
    data = result.to_dict()
    data['exported_at'] = datetime.now(timezone.utc).isoformat()
    data['original_input_included'] = include_original
    data['evidence_graph'] = build_evidence_graph(result, include_original=include_original).summary()
    if not include_original:
        for med in data['medications']:
            med['name'] = ' + '.join(med['ingredient_names']) or '[unresolved entry]'
            med['unparsed_text'] = ''
        data['unsupported_medications'] = [m['mention_id'] for m in data['medications'] if not m['normalized'] and m['status'] != 'excluded']
        rules = {r['id']: r for r in load_food_rules()[0]['rules']}
        foods = []
        for item in data['food_assessments']:
            rule = rules.get(item.get('rule_id'))
            if rule:
                prefix = {'negated': 'no ', 'uncertain': 'maybe '}.get(item['state'], '')
                item['food'] = prefix + rule['food_terms'][0]
            else:
                item['food'] = '[unmatched food entry]'
            foods.append(item['food'])
        data['foods'] = list(dict.fromkeys(foods))
        if result.hybrid_status.get('biology', {}).get('status') == 'available':
            from .biology import redacted_foods
            data['foods'] = list(dict.fromkeys(data['foods'] + redacted_foods(result.foods)))
    # Integrity check is not a digital signature; it detects accidental changes.
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    data['content_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
    return data


def render_json_report(result: AnalysisResult, *, include_original: bool = False) -> str:
    return json.dumps(export_dict(result, include_original=include_original), indent=2, ensure_ascii=False)


def replay_report(text: str, *, model_dir: str | None = None) -> AnalysisResult:
    if len(text) > 20_000_000:
        raise ValueError('Replay report exceeds 20 MB.')
    data = json.loads(text)
    checksum = data.pop('content_sha256', None)
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    if checksum != hashlib.sha256(raw.encode()).hexdigest():
        raise ValueError('Report integrity check failed.')
    if data['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported report schema.')
    current = manifest()
    for key in ('artifacts', 'code_sha256', 'algorithm'):
        if data['manifest'][key] != current[key]:
            raise ValueError(f'Replay requires the original {key}; current installation differs.')
    medications = []
    for item in data['medications']:
        item['candidates'] = tuple(Candidate(tuple(c['ingredient_ids']), tuple(c['names']), c['score'], c['method']) for c in item['candidates'])
        for key in ('ingredient_ids', 'ingredient_names', 'source_span', 'bbox'):
            if item.get(key) is not None:
                item[key] = tuple(item[key])
        medications.append(Medication(**item))
    history = []
    for item in data['review_history']:
        for key in ('ingredient_ids', 'previous_ids', 'before_findings', 'after_findings'):
            item[key] = tuple(item[key])
        history.append(ReviewDecision(**item))
    from .engine import analyze_structured
    result = analyze_structured(medications, data['foods'], history=history, run_id=data['run_id'],
                                model_dir=model_dir, **data['manifest'].get('hybrid_config', {}))
    expected = sorted((a['source_record_id'], a['severity']) for a in data['alerts'])
    actual = sorted((a.source_record_id, a.severity) for a in result.alerts)
    if expected != actual:
        raise ValueError('Replayed findings differ from the saved report.')
    if data['manifest'].get('model_artifacts') != result.manifest.get('model_artifacts'):
        raise ValueError('Replay requires the original model and calibration artifacts.')
    for field in ('candidate_assessments', 'mechanism_hypotheses', 'model_predictions'):
        if data.get(field, []) != getattr(result, field):
            raise ValueError(f'Replayed {field} differ from the saved report.')
    result.source_status['replay'] = 'Local findings reproduced. Live passages are retained only in the original report.'
    return result


def render_html_report(result: AnalysisResult, explanation: str, *, include_original: bool = False) -> str:
    from .evidence_graph import build_evidence_graph
    data = export_dict(result, include_original=include_original)
    graph = build_evidence_graph(result, include_original=include_original, include_candidates=False)
    cards = []
    for alert in result.alerts:
        links = ''.join(f'<li><a href="{escape(safe_url(e.url), quote=True)}">{escape(e.title)}</a> '
                        f'[{escape(e.evidence_id)}] — {escape(e.purpose)}<p>{escape(e.excerpt)}</p>'
                        f'<small>Record {escape(e.document_id)} · Version {escape(e.document_version)} · Retrieved {escape(e.retrieved_at)}</small></li>'
                        for e in alert.evidence)
        trace = graph.trace_finding(alert.source_record_id)
        paths = ''.join('<li>' + ' → '.join(escape(graph.graph.nodes[n]['label']) for n in path) + '</li>'
                        for path in trace['support_paths'])
        cards.append(f'<section><h3>{escape(" + ".join(alert.entities))} · {escape(alert.severity.upper())}</h3>'
                     f'<p>{escape(alert.mechanism)}</p><p>{escape(alert.management)}</p>'
                     f'<p>Record: {escape(alert.source_record_id)} · Dataset: {escape(alert.data_version)} · Entries: {escape(", ".join(alert.mention_ids))}</p>'
                     f'<p>{escape(" ".join(alert.source_disagreements))}</p><ul>{links}</ul>'
                     f'<details><summary>Recorded provenance paths</summary><ul>{paths}</ul></details></section>')
    meds = ''.join(f'<tr><td>{escape(m["mention_id"])}</td><td>{escape(m["name"])}</td><td>{escape(m["status"])}</td>'
                   f'<td>{escape(", ".join(m["ingredient_ids"]))}</td></tr>' for m in data['medications'])
    warnings = ''.join(f'<li>{escape(w)}</li>' for w in result.warnings)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Medication review dossier</title>
<style>body{{font:16px Georgia,serif;max-width:960px;margin:40px auto;padding:0 20px;color:#163334}}h1{{font-size:36px}}section{{border-left:4px solid #24756c;padding:12px 20px;background:#f1f5f2;margin:16px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ccd6d0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.notice{{padding:16px;background:#fff0d2}}@media print{{body{{margin:0}}section{{break-inside:avoid}}}}</style></head>
<body><h1>Medication review dossier</h1><p>Run {escape(result.run_id)} · Schema {SCHEMA_VERSION}</p>
<div class="notice">{DISCLAIMER}<p>Completeness: {escape(result.completeness)}</p><ul>{warnings}</ul></div>
<h2>Medication resolution</h2><table><tr><th>Entry</th><th>Medicine</th><th>Status</th><th>Ingredient IDs</th></tr>{meds}</table>
<h2>Findings</h2>{''.join(cards) or '<p>No known interaction found in this snapshot. This does not mean the combination is safe.</p>'}
<h2>Explanation</h2><pre>{escape(explanation)}</pre><h2>Pair coverage</h2><pre>{escape(json.dumps(data['pair_assessments'], indent=2))}</pre>
<h2>Candidate-dependent assessments</h2><p>Conclusions hold within retained candidates only; identities remain uncertain.</p><pre>{escape(json.dumps(data['candidate_assessments'], indent=2))}</pre>
<h2>Mechanism-supported possibilities</h2><p>Sourced biological paths do not establish clinical severity.</p><pre>{escape(json.dumps(data['mechanism_hypotheses'], indent=2))}</pre>
<h2>Experimental model predictions</h2><p>Source-label predictions are separate from documented findings.</p><pre>{escape(json.dumps(data['model_predictions'], indent=2))}</pre>
<h2>Hybrid component status</h2><pre>{escape(json.dumps(data['hybrid_status'], indent=2))}</pre>
<h2>Review history</h2><pre>{escape(json.dumps(data['review_history'], indent=2))}</pre>
<h2>Typed evidence graph</h2><pre>{escape(json.dumps(data['evidence_graph'], indent=2))}</pre>
<h2>Reproducibility manifest</h2><pre>{escape(json.dumps(data['manifest'], indent=2))}</pre>
<p>Original input included: {str(include_original).lower()}. Keep downloaded reports private when they contain sensitive information.</p></body></html>'''
