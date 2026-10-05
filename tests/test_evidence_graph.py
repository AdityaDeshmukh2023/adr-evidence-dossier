import json
from copy import deepcopy
from dataclasses import replace

import pytest

from adr_system.engine import analyze_medications, analyze_structured, apply_review
from adr_system.evidence_graph import build_evidence_graph, compare_graphs
from adr_system.explanations import explanation_payload
from adr_system.models import Evidence, Medication
from adr_system.report import render_json_report, replay_report
from adr_system.terminology import exact


def test_confirmed_finding_has_typed_paths_and_alias_invariant_ingredients():
    first = analyze_medications('Warfarin\nAspirin')
    second = analyze_medications('Warfarin\nAcetylsalicylic acid')
    graph = build_evidence_graph(first)
    alias = build_evidence_graph(second)
    assert compare_graphs(graph, alias) == {
        'introduced': [], 'removed': [], 'changed': [],
        'scope': 'Actual source finding/evidence changes, not inferred clinical outcome changes.'}
    trace = graph.trace_finding(first.alerts[0].source_record_id)
    assert trace['support_paths']
    for path in trace['support_paths']:
        assert [graph.graph.nodes[n]['kind'] for n in path] == [
            'mention', 'ingredient', 'finding', 'evidence', 'source_document']
    assert {graph.graph.nodes[p[0]]['mention_id'] for p in trace['support_paths']} == {'m1', 'm2'}


def test_candidate_record_cannot_become_a_detected_finding_without_review():
    before = analyze_medications('Warfarln\nAspirin')
    graph = build_evidence_graph(before)
    record = 'DDInter1951|DDInter20'
    assert not graph.nodes_of_kind('finding')
    assert graph.unresolved_for_record(record) == ['m1']
    assert any(r['source_record_id'] == record for p in graph.candidate_impacts('m1')['previews'] for r in p['records'])
    assert graph.evidence_for(record) == [] and not graph.trace_finding(record)['support_paths']
    after = apply_review(before, 'm1', exact('warfarin')[0].ingredient_ids)
    corrected = build_evidence_graph(after)
    assert corrected.trace_finding(record)['support_paths']
    assert compare_graphs(graph, corrected)['introduced'] == [record]
    assert corrected.review_changes('m1')[0]['introduced'] == [record]


def test_review_history_keeps_removed_records_separate_from_current_findings():
    result = analyze_medications('Warfarln\nAspirin')
    result = apply_review(result, 'm1', exact('warfarin')[0].ingredient_ids)
    before = build_evidence_graph(result)
    result = apply_review(result, 'm1', (), action='exclude_non_medication')
    after = build_evidence_graph(result)
    record = 'DDInter1951|DDInter20'
    assert compare_graphs(before, after)['removed'] == [record]
    history = after.review_changes('m1')
    assert history[0]['introduced'] == [record] and history[1]['removed'] == [record]
    assert not after.finding_nodes(record) and after.nodes_of_kind('historical_finding')


def test_identity_evidence_is_not_an_interaction_support_path_or_model_input():
    result = analyze_medications('warfarin\naspirin')
    result.alerts[0].evidence.append(Evidence('Identity', 'https://example.org/compound', 'Identity-only formula.',
                                             'PubChem', '', purpose='identity_only'))
    result.alerts[0].evidence.append(Evidence('Label passage', 'https://example.org/label', 'Supplied pair-specific passage.',
                                             'Test label', '', document_id='LABEL-1', document_version='2',
                                             purpose='supporting_label_passage'))
    graph = build_evidence_graph(result)
    record = result.alerts[0].source_record_id
    assert any(e['purpose'] == 'identity_only' for e in graph.evidence_for(record, include_identity=True))
    assert all(e['purpose'] != 'identity_only' for e in graph.evidence_for(record))
    assert 'Identity-only formula' not in json.dumps(explanation_payload(result))
    for path in graph.trace_finding(record)['support_paths']:
        assert graph.graph.nodes[path[3]]['purpose'] != 'identity_only'
    for _, target, attrs in graph.graph.edges(data=True):
        if attrs['relation'] == 'supported_by':
            graph.graph.nodes[target]['purpose'] = 'identity_only'
            break
    with pytest.raises(ValueError, match='Identity'):
        graph.validate()


def test_graph_redaction_and_report_summary_replay():
    result = analyze_medications('Patient PRIVATE_NAME\nWarfarln\nAspirin', 'PRIVATE_DRINK')
    graph = build_evidence_graph(result)
    serialized = json.dumps(graph.to_dict())
    assert 'PRIVATE_NAME' not in serialized and 'PRIVATE_DRINK' not in serialized and 'Warfarln' not in serialized
    assert 'PRIVATE_NAME' in json.dumps(build_evidence_graph(result, include_original=True).to_dict())
    report = render_json_report(result)
    assert 'PRIVATE_NAME' not in report
    assert json.loads(report)['evidence_graph']['content_sha256'] == graph.summary()['content_sha256']
    replayed = replay_report(report)
    assert compare_graphs(graph, build_evidence_graph(replayed))['changed'] == []


def test_preview_limits_are_explicit_and_do_not_remove_actual_evidence():
    result = analyze_medications('warfarin\naspirin\nWarfarln')
    graph = build_evidence_graph(result, preview_limit=0)
    assert graph.summary()['candidate_previews_truncated']
    assert not graph.candidate_impacts('m3')['complete']
    assert graph.trace_finding(result.alerts[0].source_record_id)['support_paths']
    assert not build_evidence_graph(result, include_candidates=False).candidate_impacts('m3')['complete']
    with pytest.raises(ValueError):
        build_evidence_graph(result, preview_limit=-1)


def test_alternative_candidates_are_not_merged_into_a_combination_product():
    alternatives = (replace(exact('warfarin')[0], score=.8), replace(exact('aspirin')[0], score=.7))
    unknown = Medication('uncertain', None, mention_id='m1', candidates=alternatives)
    result = analyze_structured([unknown])
    graph = build_evidence_graph(result)
    assert len(graph.nodes_of_kind('candidate')) == 2
    for candidate in graph.nodes_of_kind('candidate'):
        assert len(graph._targets(candidate, 'contains_ingredient')) == 1
    assert not graph.nodes_of_kind('finding')


def test_document_version_changes_are_visible_even_with_identical_excerpt():
    first = analyze_medications('warfarin\naspirin')
    second = deepcopy(first)
    second.alerts[0].evidence[0] = replace(second.alerts[0].evidence[0], document_version='updated-label-version')
    changes = compare_graphs(build_evidence_graph(first), build_evidence_graph(second))
    assert len(changes['changed']) == 1 and not changes['introduced'] and not changes['removed']


def test_graph_serialization_is_deterministic_and_preserves_unknown_coverage():
    result = analyze_medications('Acetaminophen\nAspirin\nZXQVVV')
    first, second = build_evidence_graph(result), build_evidence_graph(result)
    assert first.to_dict() == second.to_dict()
    assert {'unknown', 'unassessed'} <= {first.graph.nodes[n]['state'] for n in first.nodes_of_kind('coverage')}
    view, clipped = first.neighborhood(first.nodes_of_kind('mention')[0], hops=3, limit=2)
    assert len(view) <= 2 and clipped
    with pytest.raises(ValueError):
        first.candidate_impacts('missing')


def test_source_disagreements_are_queryable_without_rewriting_source_severity():
    result = analyze_medications('warfarin\naspirin')
    result.alerts[0].source_disagreements.append('Source A and source B disagree.')
    graph = build_evidence_graph(result)
    trace = graph.trace_finding(result.alerts[0].source_record_id)
    assert trace['source_disagreements'] == ['Source A and source B disagree.']
    assert graph.graph.nodes[trace['finding_nodes'][0]]['severity'] == 'high'
