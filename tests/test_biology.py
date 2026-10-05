"""Biological source and context tests; hypotheses never become clinical labels."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from adr_system.biology import (
    SNAPSHOT_PATH, mechanism_hypotheses, redacted_foods, role_feature_names,
    role_features, roles_for_ingredient, snapshot_manifest, validate_snapshot,
)
from adr_system.normalization import parse_medication_lines
from adr_system.terminology import exact


def _id(name):
    return exact(name)[0].ingredient_ids[0]


def test_clinical_paths_retain_group_and_enantiomer_conditions_without_severity():
    values = mechanism_hypotheses(parse_medication_lines('fluconazole\nwarfarin'))
    assert values and any(value['path'][0]['target'] == 'CYP2C9' for value in values)
    assert any('S-warfarin' in condition for value in values for condition in value['conditions'])
    assert all(value['status'] == 'mechanism_supported_possible' and 'severity' not in value for value in values)
    assert all(len(value['evidence']) == 2 for value in values)
    group = mechanism_hypotheses(parse_medication_lines('clarithromycin\nsimvastatin'))
    assert {value['path'][0]['target'] for value in group} == {'CYP3A', 'OATP1B1', 'OATP1B3'}
    assert not any(value['path'][0]['target'] in ('CYP3A4', 'CYP3A5') for value in group)


def test_no_uncertain_id_or_self_interaction_is_inferred():
    medications = parse_medication_lines('fluconazole\nWarfarln')
    assert not mechanism_hypotheses(medications)
    excluded = replace(parse_medication_lines('warfarin')[0], status='excluded')
    assert not mechanism_hypotheses([parse_medication_lines('fluconazole')[0], excluded])
    assert not mechanism_hypotheses(parse_medication_lines('verapamil\nverapamil'))


@pytest.mark.parametrize('text', [
    'no grapefruit juice', 'without grapefruit juice', 'grapefruit juice-free',
    'maybe grapefruit juice', 'grapefruit juice maybe', 'grapefruit juice?',
    'grapefruit juice; no grapefruit juice',
])
def test_negated_or_uncertain_exposures_do_not_create_hypotheses(text):
    assert not mechanism_hypotheses(parse_medication_lines('simvastatin'), [text])


def test_whole_product_evidence_is_not_generalized_to_constituents_or_other_foods():
    medications = parse_medication_lines('simvastatin\nrosuvastatin')
    assert not mechanism_hypotheses(medications, ['grapefruit', 'turmeric', 'citrus'])
    actual = mechanism_hypotheses(medications, ['grapefruit juice, grapefruit juice'])
    assert len([value for value in actual if value['entities'][0] == 'grapefruit juice']) == 1
    assert any(value['entities'][0] == "st. john's wort" for value in mechanism_hypotheses(medications, ["St John’s wort"]))
    assert any('preparation' in condition for value in actual for condition in value['conditions'])


def test_redacted_exposure_round_trip_keeps_semantics_without_private_context():
    medications = parse_medication_lines('simvastatin\nrosuvastatin')
    foods = ["PRIVATE PATIENT CONTEXT grapefruit juice; no St John's wort; maybe curcumin"]
    redacted = redacted_foods(foods)
    assert redacted == ['grapefruit juice', 'maybe curcumin', "no st. john's wort"]
    assert 'PRIVATE' not in json.dumps(redacted)
    assert mechanism_hypotheses(medications, foods) == mechanism_hypotheses(medications, redacted)
    assert redacted_foods(['PRIVATE unrelated intake']) == []


def test_role_features_include_documented_mask_and_unknown_is_not_negative_label():
    names = role_feature_names()
    features = role_features(_id('simvastatin'))
    assert len(names) == len(features) == 52
    assert features[0] == 1.0
    assert features[names.index('biology:substrate:CYP3A')] == 1.0
    assert not any('severity' in name or 'CYP3A4' in name for name in names)
    assert set(role_features('unknown:ingredient')) == {0.0}
    assert 'zero means not present' in snapshot_manifest()['feature_semantics']
    roles = roles_for_ingredient(_id('warfarin'))
    roles[0]['conditions'].append('caller mutation')
    assert not any('caller mutation' in role['conditions'] for role in roles_for_ingredient(_id('warfarin')))


def test_snapshot_is_source_hashed_and_unmapped_drugs_are_explicit():
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    validate_snapshot(snapshot)
    assert snapshot['source']['url'].startswith('https://www.fda.gov/')
    assert len(snapshot['source']['raw_document_sha256']) == 64
    assert snapshot['import']['unmapped_drugs'] == ['dabigatran etexilate', 'rifampin']
    assert len(snapshot['roles']) == 93
    snapshot['roles'][0]['target'] = 'CYP3A4'
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_snapshot(snapshot)


def test_importer_preserves_source_footnotes_and_rejects_missing_source_table(monkeypatch):
    from scripts import import_biology
    monkeypatch.setattr(import_biology, 'SELECTED_DRUGS', ('simvastatin',))
    monkeypatch.setattr(import_biology, 'SELECTED_EXPOSURES', {})
    header = '<tr><th>Drug or Other Substance</th>' + '<th>role</th>' * 10 + '</tr>'
    cells = ['<td>simvastatin<sup>9</sup></td>'] + ['<td></td>'] * 6
    cells += ['<td>3A sensitive substrate</td>'] + ['<td></td>'] * 3
    html = '<table>' + header + '<tr>' + ''.join(cells) + '</tr></table>'
    html += '<p><sup>9</sup>Preparation-specific test condition.</p>'
    result = import_biology.build_snapshot(html.encode(), '2026-10-04T13:00:00+00:00')
    validate_snapshot(result)
    assert result['roles'][0]['target'] == 'CYP3A'
    assert result['roles'][0]['conditions'] == ['Preparation-specific test condition.']
    with pytest.raises(ValueError, match='header not found'):
        import_biology.build_snapshot(b'<html>not the FDA table</html>', '2026-10-04')
