"""Source-backed pair screening; no model-generated interaction records."""
from __future__ import annotations

from functools import lru_cache

from .data import load_ddinter_lookup, load_interactions
from .models import Evidence, InteractionAlert
from .terminology import catalog, exact


def evidence_from(record: dict) -> list[Evidence]:
    return [Evidence(**item) for item in record.get('evidence', [])]


@lru_cache(maxsize=1)
def curated_pairs() -> dict:
    records = {}
    for key, record in load_interactions()[0]['interactions'].items():
        options = [exact(n) for n in key.split('|')]
        if all(len(x) == 1 and len(x[0].ingredient_ids) == 1 for x in options):
            ids = tuple(sorted(x[0].ingredient_ids[0] for x in options))
            records[ids] = record
    return records


def pair_alert(first: str, second: str) -> InteractionAlert | None:
    if first == second:
        return None
    ids = tuple(sorted((first, second)))
    names = catalog()[1]
    if any(i not in names for i in ids):
        raise ValueError('Unknown ingredient ID.')
    lookup, _ = load_ddinter_lookup()
    full = lookup['interactions'].get('|'.join(ids)) if lookup else None
    curated = curated_pairs().get(ids)
    if not full and not curated:
        return None
    mechanism = 'Mechanism unavailable in this source snapshot.'
    management = 'A qualified clinician or pharmacist should review the current labeling and patient context.'
    evidence, disagreements = [], []
    if full:
        version, severity = lookup['version'], full['severity']
        record_id = '|'.join(ids)
        evidence.append(Evidence('DDInter interaction record',
                                 'https://ddinter.scbdd.com/download/',
                                 f"Pair {record_id}; source severity {full['severity_source']}; categories {full['source_categories']}.",
                                 'DDInter local snapshot', '', document_id=record_id,
                                 document_version=version, section='interaction severity'))
    else:
        version = load_interactions()[0]['version']
        severity, record_id = curated['severity'], curated['id']
    if curated:
        mechanism, management = curated['mechanism'], curated['management']
        evidence.extend(evidence_from(curated))
        if full and full['severity'] != curated['severity']:
            disagreements.append(f"Full snapshot: {full['severity']}; curated demonstration: {curated['severity']}.")
    return InteractionAlert([names[i] for i in ids], 'drug-drug', severity, mechanism,
                            management, evidence, version, None, record_id,
                            ingredient_ids=list(ids), source_disagreements=disagreements)


@lru_cache(maxsize=16384)
def pair_state(first: str, second: str) -> str:
    if first == second:
        return 'same_ingredient'
    alert = pair_alert(first, second)
    return alert.severity if alert else 'no_record'
