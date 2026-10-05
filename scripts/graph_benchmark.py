"""Synthetic graph traceability/conformance experiment; no clinical benefit score."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_figure(before, after, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from textwrap import fill
    record = 'DDInter1951|DDInter20'
    preview = next(p for p in before.candidate_impacts('m1')['previews']
                   if any(r['source_record_id'] == record for r in p['records']))
    candidate = before._targets(preview['node_id'], 'assumes_candidate')[0]
    mention = next(n for n in before.nodes_of_kind('mention') if before.graph.nodes[n]['mention_id'] == 'm1')
    source = next(r['node_id'] for r in preview['records'] if r['source_record_id'] == record)
    trace = after.trace_finding(record)
    path = next(p for p in trace['support_paths'] if after.graph.nodes[p[0]]['mention_id'] == 'm1'
                and after.graph.nodes[p[-1]]['document_id'] == record)
    fig, axes = plt.subplots(1, 2, figsize=(11, 8))
    panels = [(before, [mention, candidate, preview['node_id'], source],
               [(0, 1, 'has candidate'), (2, 1, 'assumes candidate'), (2, 3, 'could retrieve')],
               'Before review: candidate-dependent possibility'),
              (after, path, [(0, 1, 'resolved to'), (1, 2, 'participates in'),
                             (2, 3, 'supported by'), (3, 4, 'from document')],
               'After review: recorded provenance path')]
    for ax, (graph, nodes, edges, title) in zip(axes, panels):
        ys = [0.88 - i * .18 for i in range(len(nodes))]
        for i, ident in enumerate(nodes):
            data = graph.graph.nodes[ident]
            label = data['label']
            if data['kind'] == 'source_document':
                label = data['source'] + '\n' + data['document_version']
            elif data['kind'] == 'candidate_preview':
                label = 'Candidate-dependent scenario'
            elif data['kind'] == 'source_record':
                label = 'Possible source record\n' + record
            elif data['kind'] == 'finding':
                label = 'Detected source finding\n' + record
            label = data['kind'].replace('_', ' ').upper() + '\n' + '\n'.join(fill(line, 32) for line in label.splitlines())
            ax.text(.46, ys[i], label, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=.55', facecolor='#fff0d2' if ax is axes[0] else '#edf2ee',
                              edgecolor='#b98124' if ax is axes[0] else '#24756c'))
        for first, second, relation in edges:
            downward = ys[first] > ys[second]
            start = ys[first] + (-.055 if downward else .055)
            end = ys[second] + (.055 if downward else -.055)
            ax.annotate('', xy=(.46, end), xytext=(.46, start),
                        arrowprops=dict(arrowstyle='->', color='#b98124' if ax is axes[0] else '#24756c',
                                        linestyle='--' if ax is axes[0] else '-', lw=1.5))
            ax.text(.50, (ys[first] + ys[second]) / 2, relation, va='center', fontsize=9)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title)
        ax.axis('off')
    fig.suptitle('Typed evidence graph: identity review and source traceability', fontsize=15, y=.98)
    fig.text(.5, .025, 'Synthetic warfarin/aspirin example. Dashed relationships are hypothetical; paths do not establish patient-specific risk.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(output / 'graph_review_transition.svg')
    fig.savefig(output / 'graph_review_transition.png', dpi=220)
    plt.close(fig)


def main():
    from adr_system.engine import analyze_medications, apply_review
    from adr_system.evidence_graph import build_evidence_graph, compare_graphs
    from adr_system.provenance import manifest
    from adr_system.terminology import exact

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/evidence_graph')
    args = parser.parse_args()
    fixtures = json.loads((ROOT / 'data/evaluation_v2.json').read_text(encoding='utf-8'))
    scenarios = [(case['id'], analyze_medications(case['medications'], case['foods'])) for case in fixtures['cases']]
    before = analyze_medications('Warfarln\nAspirin')
    before.run_id = 'GRAPH-REVIEW-DEMO'
    after = apply_review(before, 'm1', exact('warfarin')[0].ingredient_ids)
    excluded = apply_review(after, 'm1', (), action='exclude_non_medication')
    scenarios.extend([('review_before', before), ('review_after', after), ('review_excluded', excluded)])
    args.output.mkdir(parents=True, exist_ok=True)
    rows, paths = [], []
    for scenario, result in scenarios:
        if not scenario.startswith('review_'):
            result.run_id = f'GRAPH-{scenario}'
        start = perf_counter()
        graph = build_evidence_graph(result)
        build_ms = (perf_counter() - start) * 1000
        traceable, supports, identifiers, equal_sets = 0, 0, 0, True
        times = []
        for alert in result.alerts:
            start = perf_counter()
            trace = graph.trace_finding(alert.source_record_id)
            times.append((perf_counter() - start) * 1000)
            traceable += bool(trace['support_paths'])
            expected = {(e.evidence_id, e.excerpt) for e in alert.evidence if e.purpose != 'identity_only'}
            actual = {(e['evidence_id'], e['excerpt']) for e in trace['evidence']}
            equal_sets &= expected == actual
            for evidence in trace['evidence']:
                supports += 1
                identifiers += any(graph.graph.nodes[n]['identifier_available']
                                   for n in graph._targets(evidence['node_id'], 'from_document'))
            for path in trace['support_paths']:
                paths.append({'scenario': scenario, 'record_id': alert.source_record_id,
                              'node_ids': path, 'node_types': [graph.graph.nodes[n]['kind'] for n in path],
                              'labels': [graph.graph.nodes[n]['label'] for n in path]})
        summary = graph.summary()
        rows.append({'scenario': scenario, 'nodes': summary['nodes'], 'edges': summary['edges'],
                     'detected_findings': len(result.alerts), 'structurally_traceable_findings': traceable,
                     'support_passages': supports, 'passages_with_document_identifier': identifiers,
                     'same_evidence_as_flat_records': equal_sets,
                     'candidate_previews_truncated': summary['candidate_previews_truncated'],
                     'build_ms': build_ms, 'trace_query_median_ms': statistics.median(times) if times else None})
        if scenario in ('review_before', 'review_after'):
            (args.output / f'{scenario}.graph.json').write_text(json.dumps(graph.to_dict(), indent=2), encoding='utf-8')
    corrected = build_evidence_graph(after)
    write_figure(build_evidence_graph(before), corrected, args.output)
    transitions = {'correction': compare_graphs(build_evidence_graph(before), corrected),
                   'exclusion': compare_graphs(corrected, build_evidence_graph(excluded)),
                   'review_history': build_evidence_graph(excluded).review_changes('m1')}
    passed = all(r['same_evidence_as_flat_records'] and
                 r['structurally_traceable_findings'] == r['detected_findings'] for r in rows)
    passed &= bool(transitions['correction']['introduced']) and bool(transitions['exclusion']['removed'])
    summary = {'dataset_kind': 'synthetic', 'passed': passed, 'scenarios': len(rows),
               'detected_findings': sum(r['detected_findings'] for r in rows),
               'structurally_traceable_findings': sum(r['structurally_traceable_findings'] for r in rows),
               'graph_flat_evidence_equivalence': all(r['same_evidence_as_flat_records'] for r in rows),
               'build_median_ms': statistics.median(r['build_ms'] for r in rows),
               'support_paths': len(paths), 'manifest': manifest(),
               'scope': 'Graph structure, provenance reachability and evidence-set conformance on software fixtures.',
               'limitations': ['Trace reachability does not establish clinical relevance or citation quality.',
                               'The scenarios are fixtures and review transitions, not independent patient prescriptions.',
                               'Graph/flat equivalence preserves source outputs; no accuracy improvement is claimed.',
                               'No human review-time, real-image or clinical outcome benefit was measured.',
                               'No external model or evidence requests were made.']}
    with (args.output / 'case_metrics.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for filename, data in [('summary.json', summary), ('support_paths.json', paths), ('review_changes.json', transitions)]:
        (args.output / filename).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('passed', 'scenarios', 'detected_findings',
                                                   'structurally_traceable_findings', 'graph_flat_evidence_equivalence')}, indent=2))
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
