import json
from dataclasses import replace

import pytest

from adr_system.data import load_ddinter_lookup
from adr_system.engine import analyze_medications, analyze_structured, apply_review
from adr_system.explanations import bounded_llm_explanation, explanation_payload, validate_claims
from adr_system.models import Candidate
from adr_system.normalization import normalise_medication, parse_medication_lines
from adr_system.report import render_html_report, render_json_report, replay_report
from adr_system.terminology import exact
from paddle_ocr import medications_from_document, parse_ocr_results


def test_all_aspirin_source_pairs_are_accessible():
    lookup, _ = load_ddinter_lookup()
    ident = exact('aspirin')[0].ingredient_ids[0]
    assert ident == exact('acetylsalicylic acid')[0].ingredient_ids[0]
    from adr_system.knowledge import pair_alert
    rows = [r for r in lookup['interactions'].values() if ident in (r['drug_a_id'], r['drug_b_id'])]
    assert len(rows) == 654
    for row in rows:
        result = pair_alert(row['drug_a_id'], row['drug_b_id'])
        assert result and result.severity == row['severity']


@pytest.mark.parametrize('text', ['Warfarin 5 mg Aspirin 75 mg', 'Tab. Warfarin 5 mg\nAspirin 75 mg', 'warfarin daily\naspirin'])
def test_no_medication_loss(text):
    result = analyze_medications(text)
    assert len(result.medications) == 2
    assert len(result.alerts) == 1
    assert result.alerts[0].severity == 'high'


def test_attributes_and_leftover_text_are_preserved():
    result = analyze_medications('Warfarin 5 mg oral daily mysterious medicine')
    assert any(m.ingredient_ids for m in result.medications)
    assert result.completeness == 'incomplete'
    assert any('mysterious' in m.name for m in result.medications)
    med = normalise_medication('Warfarin 5 mg oral daily')
    assert (med.strength, med.route, med.frequency) == ('5 mg', 'oral', 'daily')


def test_all_catalog_names_resolve_without_attribute_prefix_damage():
    from adr_system.terminology import catalog
    for name, choices in catalog()[0].items():
        if len(choices) == 1:
            assert normalise_medication(name).ingredient_ids == choices[0].ingredient_ids, name


@pytest.mark.parametrize('food', ['no grapefruit', 'without grapefruit', 'grapefruit-free'])
def test_food_negation(food):
    result = analyze_medications('simvastatin', food)
    assert not result.alerts
    assert result.food_assessments[0]['state'] == 'negated'


def test_food_dedup_and_boundaries():
    assert not analyze_medications('warfarin', 'kaleidoscope').alerts
    assert len(analyze_medications('simvastatin', 'grapefruit, grapefruit juice').alerts) == 1
    uncertain = analyze_medications('simvastatin', 'maybe grapefruit')
    assert not uncertain.alerts
    assert uncertain.food_assessments[0]['state'] == 'uncertain'


def test_unknown_is_not_no_record():
    result = analyze_medications('acetaminophen\naspirin')
    assert result.alerts[0].severity == 'unknown'
    assert any('Unknown does not mean' in w for w in result.warnings)


def test_review_resolves_without_rewriting_original():
    original = analyze_medications('Warfarln 5 mg\nAspirin\nMetformln 500 mg')
    assert not original.alerts
    assert original.review_queue[0].mention_id == 'm1'
    updated = apply_review(original, 'm1', exact('warfarin')[0].ingredient_ids)
    assert updated.alerts[0].severity == 'high'
    assert updated.medications[0].name == 'Warfarln 5 mg'
    assert not original.alerts
    assert updated.review_history[-1].after_findings != updated.review_history[-1].before_findings
    assert updated.completeness == 'incomplete'


def test_no_candidate_requires_manual_review():
    result = analyze_medications('ZXQVVV\nAspirin')
    assert result.review_queue[0].required
    assert result.pair_assessments[0]['state'] == 'unassessed'
    updated = apply_review(result, 'm1', (), action='exclude_non_medication')
    assert updated.medications[0].status == 'excluded'


def test_candidate_generation_does_not_use_partner():
    first = analyze_medications('Warfarln\nAspirin').medications[0]
    second = analyze_medications('Warfarln\nMetformin').medications[0]
    assert first.candidates == second.candidates
    assert not first.ingredient_ids


def test_combination_ingredient_set_and_duplicate_owner():
    a, b = exact('warfarin')[0], exact('aspirin')[0]
    med = replace(normalise_medication('warfarin'), mention_id='combo',
                  ingredient_ids=a.ingredient_ids + b.ingredient_ids,
                  ingredient_names=a.names + b.names)
    result = analyze_structured([med, replace(normalise_medication('aspirin'), mention_id='extra')])
    assert len(result.alerts) == 1
    assert any('Repeated ingredient' in w for w in result.warnings)


def test_ocr_v3_dictionary_and_geometry():
    lines = parse_ocr_results([{'rec_texts': ['Warfarin 5 mg', 'Aspirin 75 mg'],
                              'rec_scores': [.99, .70], 'rec_boxes': [[1, 2, 60, 20], [1, 25, 60, 40]]}])
    meds = medications_from_document({'lines': lines})
    assert meds[0].canonical_name == 'warfarin'
    assert meds[0].bbox == (1., 2., 60., 20.)
    assert not meds[1].normalized and meds[1].candidates
    assert not meds[1].ingredient_ids


def test_export_redaction_integrity_and_replay():
    result = analyze_medications('Patient PRIVATE_NAME\nWarfarln\nAspirin', 'grapefruit juice')
    result = apply_review(result, 'm2', exact('warfarin')[0].ingredient_ids)
    serialized = render_json_report(result)
    assert 'PRIVATE_NAME' not in serialized and 'Warfarln' not in serialized
    replayed = replay_report(serialized)
    assert [(a.source_record_id, a.severity) for a in result.alerts] == [(a.source_record_id, a.severity) for a in replayed.alerts]
    assert replayed.completeness == result.completeness
    assert replayed.review_history == result.review_history
    tampered = json.loads(serialized)
    tampered['run_id'] = 'changed'
    with pytest.raises(ValueError, match='integrity'):
        replay_report(json.dumps(tampered))


def test_report_includes_unresolved_warnings_and_escapes_input():
    result = analyze_medications('<script>alert(1)</script>\nWarfarin')
    report = render_html_report(result, '<script>bad</script>', include_original=True)
    assert '<script>' not in report
    assert 'Assessment is incomplete' in report


def test_external_payload_excludes_raw_input_and_disabled_means_no_call(monkeypatch):
    result = analyze_medications('Patient PRIVATE_NAME\nWarfarin\nAspirin')
    assert 'PRIVATE_NAME' not in json.dumps(explanation_payload(result))
    def fail(*args, **kwargs):
        raise AssertionError('External call attempted')
    monkeypatch.setattr('adr_system.explanations._generate', fail)
    text, mode = bounded_llm_explanation(result)
    assert 'disabled' in mode


def test_claim_validation_rejects_inventions():
    result = analyze_medications('warfarin\naspirin')
    payload = explanation_payload(result)
    record = payload['records'][0]
    evidence = record['evidence'][0]
    claim = {'record_id': record['record_id'], 'severity': record['severity'],
             'evidence_ids': [evidence['id']], 'quote': evidence['excerpt']}
    assert validate_claims(json.dumps({'claims': [claim]}), payload)
    claim['quote'] = 'Stop taking your medication.'
    with pytest.raises(ValueError):
        validate_claims(json.dumps({'claims': [claim]}), payload)


def test_evidence_attachment_by_id_and_refresh_deduplicates(monkeypatch):
    from adr_system.models import Evidence
    from adr_system.evidence import enrich_live
    result = analyze_medications('Abacavir\nNaltrexone')
    ident = exact('abacavir')[0].ingredient_ids[0]
    evidence = Evidence('label', 'https://example.org/label', 'Naltrexone passage.', 'openFDA label', '2026-10-03', purpose='supporting_label_passage')
    monkeypatch.setattr('adr_system.evidence.live_evidence_for', lambda drugs: ({ident: [evidence]}, {'openFDA:abacavir': 'live'}))
    enrich_live(result)
    enrich_live(result)
    assert sum(e.evidence_id == evidence.evidence_id for e in result.alerts[0].evidence) == 1


def test_bad_inputs_and_empty_results():
    assert analyze_medications('').completeness == 'empty'
    with pytest.raises(ValueError):
        parse_medication_lines('x' * 10001)
    with pytest.raises(ValueError):
        parse_medication_lines('\n'.join(['aspirin'] * 31))
    with pytest.raises(ValueError):
        apply_review(analyze_medications('aspirin'), 'm1', ('unknown-id',))
