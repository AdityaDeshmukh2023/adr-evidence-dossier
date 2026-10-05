import json
from types import SimpleNamespace

import pytest

from adr_system.engine import analyze_medications
from adr_system.explanations import (
    DEFAULT_GROQ_MODEL, PROMPT_VERSION, _generate, bounded_llm_explanation,
    configured_model, explanation_payload, validate_claims,
)


def valid_document(payload):
    return {'claims': [{'record_id': record['record_id'], 'severity': record['severity'],
                        'evidence_ids': [record['evidence'][0]['id']],
                        'quote': record['evidence'][0]['excerpt']}
                       for record in payload['records']]}


def test_groq_request_is_bounded_and_validates_source_contract(monkeypatch):
    groq = pytest.importorskip('groq', reason='Optional Groq profile is not installed.')
    calls = []
    result = analyze_medications('Patient PRIVATE_NAME\nWarfarin\nAspirin')
    payload = explanation_payload(result)

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=json.dumps(valid_document(payload))))],
            usage=SimpleNamespace(model_dump=lambda: {'prompt_tokens': 1000, 'completion_tokens': 200}))

    def client(**kwargs):
        assert kwargs['timeout'] == 20 and kwargs['max_retries'] == 1
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(groq, 'Groq', client)
    monkeypatch.setenv('GROQ_API_KEY', 'synthetic-test-credential')
    monkeypatch.delenv('GROQ_MODEL', raising=False)
    _generate.cache_clear()
    try:
        text, mode = bounded_llm_explanation(result, enabled=True)
        assert mode == 'validated extractive Groq explanation'
        assert 'Selected source excerpts' in text
        request = calls[0]
        assert request['model'] == DEFAULT_GROQ_MODEL
        assert request['max_completion_tokens'] == 1600
        assert request['reasoning_effort'] == 'low' and request['include_reasoning'] is False
        assert 'PRIVATE_NAME' not in json.dumps(request)
        schema = request['response_format']['json_schema']
        assert schema['strict'] is True
        assert schema['schema']['additionalProperties'] is False
        assert result.manifest['explanation']['estimated_cost_usd'] == pytest.approx(.000135)
        bounded_llm_explanation(result, enabled=True)
        assert len(calls) == 1
        bounded_llm_explanation(result, enabled=False)
        assert 'explanation' not in result.manifest
    finally:
        _generate.cache_clear()


def test_truncated_model_response_uses_deterministic_fallback(monkeypatch):
    groq = pytest.importorskip('groq', reason='Optional Groq profile is not installed.')
    monkeypatch.setenv('GROQ_API_KEY', 'synthetic-test-credential')
    monkeypatch.delenv('GROQ_MODEL', raising=False)
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='length')])
    monkeypatch.setattr(groq, 'Groq', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    _generate.cache_clear()
    try:
        result = analyze_medications('warfarin\naspirin')
        text, mode = bounded_llm_explanation(result, enabled=True)
        assert 'fallback' in mode
        assert 'Selected source excerpts' not in text and 'explanation' not in result.manifest
    finally:
        _generate.cache_clear()


@pytest.mark.parametrize('document', [[], {'claims': [], 'advice': 'invented'}, {'claims': [None]}])
def test_invalid_claim_documents_are_rejected(document):
    payload = explanation_payload(analyze_medications('warfarin\naspirin'))
    with pytest.raises(ValueError):
        validate_claims(json.dumps(document), payload)


def test_empty_model_configuration_uses_current_default(monkeypatch):
    monkeypatch.setenv('GROQ_MODEL', '  ')
    assert configured_model() == DEFAULT_GROQ_MODEL
