"""Integration boundaries for candidates, biology, optional models and exports."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from adr_system import biology
from adr_system.engine import analyze_medications
from adr_system.evidence_graph import build_evidence_graph
from adr_system.explanations import explanation_payload, validate_claims
from adr_system.report import render_json_report, replay_report

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'llm.py'
MISSING_MODEL = ROOT / 'artifacts' / '__integration_missing_model__'


def _source_signature(result):
    return sorted((alert.source_record_id, alert.severity,
                   tuple(e.evidence_id for e in alert.evidence)) for alert in result.alerts)


def test_stable_candidate_state_cannot_become_a_documented_finding_or_groq_claim():
    result = analyze_medications('Warfarln\nAspirin')
    assessment = result.candidate_assessments[0]
    assert assessment['status'] == 'stable' and assessment['stable_states'] == ['high']
    assert assessment['stable_record_ids']
    assert result.alerts == []
    graph = build_evidence_graph(result)
    assert graph.nodes_of_kind('candidate_assessment')
    assert not graph.nodes_of_kind('finding')
    for record in assessment['stable_record_ids']:
        assert not graph.evidence_for(record)
        assert not graph.trace_finding(record)['support_paths']
    payload = explanation_payload(result)
    assert payload['records'] == []
    forged = {'claims': [{'record_id': assessment['stable_record_ids'][0],
                          'severity': 'high', 'evidence_ids': ['candidate-evidence'],
                          'quote': 'A candidate preview is not clinical evidence.'}]}
    with pytest.raises(ValueError, match='Missing or extra'):
        validate_claims(json.dumps(forged), payload)


def test_biology_and_model_rationale_never_enter_documented_support_paths_or_groq():
    baseline = analyze_medications('clarithromycin\nsimvastatin', include_hybrid=False)
    result = analyze_medications('clarithromycin\nsimvastatin')
    assert result.mechanism_hypotheses and result.alerts
    assert _source_signature(result) == _source_signature(baseline)
    ids = result.alerts[0].ingredient_ids
    # An adversarial model output must remain separate even if it contradicts a source.
    result.model_predictions.append({
        'ingredient_ids': ids, 'predicted_severity': 'low', 'accepted': True,
        'model_version': 'integration-stub', 'explanation_reference': 'MODEL_RATIONALE_PRIVATE_SENTINEL',
    })
    graph = build_evidence_graph(result)
    assert graph.nodes_of_kind('mechanism_hypothesis')
    assert graph.nodes_of_kind('biological_role')
    assert graph.nodes_of_kind('model_prediction') and graph.nodes_of_kind('model_rationale')
    biological_sources = {
        target for _, target, attrs in graph.graph.edges(data=True)
        if attrs['relation'] == 'role_source'}
    assert biological_sources
    for alert in result.alerts:
        trace = graph.trace_finding(alert.source_record_id)
        assert trace['support_paths']
        for path in trace['support_paths']:
            assert [graph.graph.nodes[node]['kind'] for node in path] == [
                'mention', 'ingredient', 'finding', 'evidence', 'source_document']
            assert not biological_sources.intersection(path)
    for source, target, attrs in graph.graph.edges(data=True):
        if attrs['relation'] == 'supported_by':
            assert graph.graph.nodes[source]['kind'] == 'finding'
            assert graph.graph.nodes[target]['kind'] == 'evidence'
    payload = explanation_payload(result)
    assert payload == explanation_payload(baseline)
    assert 'MODEL_RATIONALE_PRIVATE_SENTINEL' not in json.dumps(payload)
    assert 'fda.gov/drugs/drug-interactions-labeling' not in json.dumps(payload)


def test_missing_optional_model_preserves_known_and_unknown_source_severities():
    baseline = analyze_medications('warfarin\naspirin\nacetaminophen')
    enabled = analyze_medications('warfarin\naspirin\nacetaminophen',
                                  enable_research_model=True, model_dir=str(MISSING_MODEL))
    assert any(alert.severity == 'unknown' for alert in baseline.alerts)
    assert any(alert.severity == 'high' for alert in baseline.alerts)
    assert _source_signature(enabled) == _source_signature(baseline)
    assert enabled.model_predictions
    eligible = {alert.source_record_id for alert in baseline.alerts if alert.severity == 'unknown'}
    assert {prediction['source_record_id'] for prediction in enabled.model_predictions} == eligible
    assert all(not prediction['accepted'] and prediction['predicted_severity'] is None
               and prediction['probabilities'] == {}
               and prediction['abstention_reason'] == 'model_artifacts_absent'
               for prediction in enabled.model_predictions)
    assert enabled.hybrid_status['model']['status'] not in ('ready', 'available', 'disabled')
    assert explanation_payload(enabled) == explanation_payload(baseline)


@pytest.mark.parametrize('meds, foods', [
    ('simvastatin', "PRIVATE DIET HEADER grapefruit juice; no St John's wort; maybe curcumin"),
    ('simvastatin', "no grapefruit juice but St John's wort"),
    ('simvastatin', 'grapefruit juice; no grapefruit juice'),
    ('warfarin\nsimvastatin', 'PRIVATE DIET HEADER spinach and grapefruit juice'),
    ('rosuvastatin', 'PRIVATE DIET HEADER curcumin; no diosmin'),
])
def test_fda_food_redaction_replays_hypotheses_and_original_source_findings(meds, foods):
    result = analyze_medications(meds, foods)
    encoded = render_json_report(result)
    assert 'private diet header' not in encoded.casefold()
    exported = json.loads(encoded)
    assert all('private diet header' not in item.casefold() for item in exported['foods'])
    reproduced = replay_report(encoded)
    assert reproduced.mechanism_hypotheses == result.mechanism_hypotheses
    assert reproduced.candidate_assessments == result.candidate_assessments
    assert _source_signature(reproduced) == _source_signature(result)


@pytest.mark.parametrize('failure', ['corrupt', 'missing'])
def test_bad_biology_fails_closed_while_documented_screening_and_export_survive(monkeypatch, failure):
    baseline = analyze_medications('warfarin\naspirin')
    if failure == 'corrupt':
        snapshot = json.loads(biology.SNAPSHOT_PATH.read_text(encoding='utf-8'))
        snapshot['roles'][0]['target'] = 'CYP3A4'
        read_bytes = lambda: json.dumps(snapshot).encode()
    else:
        def read_bytes():
            raise FileNotFoundError('Biology snapshot is missing.')
    monkeypatch.setattr(biology, 'SNAPSHOT_PATH', SimpleNamespace(read_bytes=read_bytes))
    biology._snapshot.cache_clear()
    biology._role_index.cache_clear()
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            biology.snapshot_manifest()
        result = analyze_medications('warfarin\naspirin')
        assert _source_signature(result) == _source_signature(baseline)
        assert not result.mechanism_hypotheses
        assert result.hybrid_status['biology']['status'] in ('incompatible', 'unavailable')
        encoded = render_json_report(result)
        assert not json.loads(encoded)['mechanism_hypotheses']
    finally:
        biology._snapshot.cache_clear()
        biology._role_index.cache_clear()


def test_hybrid_ui_examples_candidate_panel_and_stale_exports(monkeypatch):
    monkeypatch.setenv('HYBRID_MODEL_DIR', str(MISSING_MODEL))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.selectbox(key='case_choice').select('06 / Biological pathway').run()
    next(button for button in app.button if button.label == 'Build review dossier').click().run()
    assert not app.exception
    result = app.session_state['result']
    assert result.mechanism_hypotheses
    assert result.hybrid_status['model']['status'] == 'disabled'
    assert {'Uncertain identities', 'Biological paths', 'Model predictions'} <= {tab.label for tab in app.tabs}
    assert len(app.get('download_button')) == 2
    # Model configuration is part of the dossier input; old exports must disappear.
    app.checkbox(key='enable_local_model').set_value(True).run()
    assert not app.exception
    assert any('input has changed' in warning.value for warning in app.warning)
    assert not app.get('download_button')
    next(button for button in app.button if button.label == 'Build review dossier').click().run()
    assert not app.exception
    assert app.session_state['result'].hybrid_status['model']['status'] not in ('ready', 'available', 'disabled')
    assert len(app.get('download_button')) == 2
    app.selectbox(key='case_choice').select('02 / Ambiguous medicine reading').run()
    next(button for button in app.button if button.label == 'Build review dossier').click().run()
    assert not app.exception
    assert any(item['status'] != 'resolved' for item in app.session_state['result'].candidate_assessments)
    assert any(expander.label == 'Candidate identities and supporting record IDs' for expander in app.expander)
    app.text_area(key='food_input').set_value('PRIVATE changed dietary text').run()
    assert not app.exception
    assert any('input has changed' in warning.value for warning in app.warning)
    assert not app.get('download_button')
    app.radio[0].set_value('Interaction map').run()
    assert not app.exception
    assert any('input has changed' in warning.value for warning in app.warning)
    assert not app.get('download_button')
