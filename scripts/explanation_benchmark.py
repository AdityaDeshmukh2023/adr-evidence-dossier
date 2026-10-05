"""Generate blinded explanation-rating sheets. Paid calls require --live."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    from adr_system.engine import analyze_medications
    from adr_system.explanations import (PROMPT_VERSION, deterministic_explanation, explanation_payload,
                                        bounded_llm_explanation, configured_model, model_options, estimated_cost)
    from adr_system.provenance import manifest
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--max-cases', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/explanations')
    args = parser.parse_args()
    if not 1 <= args.max_cases <= 10:
        raise ValueError('Use 1–10 cases to bound API requests.')
    if args.live and not os.getenv('GROQ_API_KEY'):
        raise ValueError('GROQ_API_KEY is required for --live; no requests were sent.')
    args.output.mkdir(parents=True, exist_ok=True)
    examples = ['warfarin\naspirin', 'abacavir\nnaltrexone', 'fluoxetine\ntramadol', 'clopidogrel\nwarfarin'][:args.max_cases]
    outputs, ratings = [], []
    for i, text in enumerate(examples):
        result = analyze_medications(text)
        payload = explanation_payload(result)
        for method in ('template', 'prompt_only', 'validated_extractive'):
            if method != 'template' and not args.live:
                continue
            start = perf_counter()
            metadata = {}
            if method == 'template':
                explanation, mode = deterministic_explanation(result), 'template'
            elif method == 'validated_extractive':
                explanation, mode = bounded_llm_explanation(result, enabled=True)
                metadata = result.manifest.get('explanation', {})
            else:
                from groq import Groq
                model = configured_model()
                response = Groq(timeout=20, max_retries=1).chat.completions.create(model=model, temperature=0,
                    max_completion_tokens=1600, **model_options(model),
                    messages=[{'role': 'system', 'content': 'Explain only the supplied medication interaction records. Cite their evidence IDs. Do not invent facts or recommend treatment changes.'},
                              {'role': 'user', 'content': json.dumps(payload)}])
                explanation, mode = response.choices[0].message.content, 'unvalidated experimental baseline; never shown as app guidance'
                usage = response.usage.model_dump() if response.usage else {}
                metadata = {'model': model, 'usage': usage, 'finish_reason': response.choices[0].finish_reason,
                            **model_options(model), **estimated_cost(usage, model)}
            identity = hashlib.sha256(f'{i}|{method}'.encode()).hexdigest()[:12]
            outputs.append({'output_id': identity, 'case_id': f'EXPL-{i + 1}', 'method': method,
                            'text': explanation, 'mode': mode, 'evidence': payload, 'metadata': metadata,
                            'latency_ms': (perf_counter() - start) * 1000})
            ratings.append({'output_id': identity, 'supported_claims': '', 'total_claims': '',
                            'correct_citations': '', 'total_citations': '', 'omissions': '', 'reviewer_id': '', 'notes': ''})
    (args.output / 'outputs.json').write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    blinded = [{'output_id': o['output_id'], 'text': o['text'], 'evidence': o['evidence']} for o in outputs]
    blinded.sort(key=lambda x: x['output_id'])
    (args.output / 'blinded_review.json').write_text(json.dumps(blinded, indent=2), encoding='utf-8')
    with (args.output / 'ratings.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(ratings[0])); writer.writeheader(); writer.writerows(ratings)
    extracted = [o for o in outputs if o['method'] == 'validated_extractive']
    summary = {'outputs': len(outputs), 'live_requested': args.live,
               'model': configured_model() if args.live else None, 'prompt_version': PROMPT_VERSION,
               'validated_outputs': sum(o['mode'] == 'validated extractive Groq explanation' for o in extracted),
               'extractive_attempts': len(extracted),
               'estimated_cost_usd': round(sum(o['metadata'].get('estimated_cost_usd') or 0 for o in outputs), 9) if args.live else 0,
               'manifest': manifest(),
               'unsupported_claim_rate': None, 'citation_correctness': None,
               'status': 'Expert claim-level ratings pending. No clinical faithfulness score has been fabricated.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
