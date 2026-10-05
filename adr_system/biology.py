"""Offline, evidence-backed mechanism hypotheses; no clinical pair severity inference."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from .models import Medication
from .terminology import catalog

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'biological_roles.json'
TARGETS = ('CYP1A2', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A',
           'P-gp', 'BCRP', 'OATP1B', 'OATP1B1', 'OATP1B3', 'OAT1', 'OAT3',
           'OCT2', 'MATE1', 'MATE2-K')
ROLES = ('inhibitor', 'inducer', 'substrate')


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_snapshot(snapshot: dict) -> None:
    """Fail closed if the frozen role facts, mappings, or extracted rows are altered."""
    if snapshot.get('schema_version') != '1.0' or not snapshot.get('roles'):
        raise ValueError('Unsupported or empty biology snapshot.')
    if snapshot.get('roles_sha256') != _canonical_hash(snapshot['roles']):
        raise ValueError('Biology role content hash mismatch.')
    rows = snapshot.get('selected_rows', [])
    if snapshot['source'].get('selected_rows_sha256') != _canonical_hash(rows):
        raise ValueError('Biology selected source rows hash mismatch.')
    if not re.fullmatch(r'[0-9a-f]{64}', snapshot['source'].get('raw_document_sha256', '')):
        raise ValueError('Missing raw source document hash.')
    row_hashes = set()
    for row in rows:
        if row['row_sha256'] != _canonical_hash(row['cells']):
            raise ValueError('Biology source row hash mismatch.')
        row_hashes.add(row['row_sha256'])
    ids = set()
    mappings = snapshot['import']['ingredient_mappings']
    for role in snapshot['roles']:
        if role['role_id'] in ids:
            raise ValueError('Duplicate biology role ID.')
        ids.add(role['role_id'])
        if role['target'] not in TARGETS or role['role'] not in ROLES:
            raise ValueError('Unsupported biology role target or relation.')
        if role['source_row_sha256'] not in row_hashes or role['source_url'] != snapshot['source']['url']:
            raise ValueError('Unlinked biological source fact.')
        if len(role['footnote_ids']) != len(role['conditions']):
            raise ValueError('Missing biological source condition.')
        ident = role['ingredient_id']
        if ident and mappings.get(role['source_entity']) != ident:
            raise ValueError('Biology ingredient mapping mismatch.')


@lru_cache(maxsize=1)
def _snapshot() -> tuple[dict, str]:
    raw = SNAPSHOT_PATH.read_bytes()
    value = json.loads(raw)
    validate_snapshot(value)
    return value, hashlib.sha256(raw).hexdigest()


def snapshot_manifest() -> dict:
    value, digest = _snapshot()
    return {'version': value['version'], 'sha256': digest,
            'roles_sha256': value['roles_sha256'], 'source': dict(value['source']),
            'role_count': len(value['roles']),
            'mapped_drug_count': len(value['import']['ingredient_mappings']),
            'unmapped_drugs': list(value['import']['unmapped_drugs']),
            'limitations': list(value['limitations']),
            'feature_semantics': 'binary documented role presence; zero means not present in bounded snapshot'}


@lru_cache(maxsize=1)
def _role_index() -> dict[str, tuple[dict, ...]]:
    grouped: dict[str, list[dict]] = {}
    for role in _snapshot()[0]['roles']:
        if role['ingredient_id']:
            grouped.setdefault(role['ingredient_id'], []).append(role)
    return {ident: tuple(items) for ident, items in grouped.items()}


def roles_for_ingredient(ingredient_id: str) -> tuple[dict, ...]:
    """Only mapped ingredient IDs are accepted; no fuzzy name or severity matching."""
    return tuple(deepcopy(role) for role in _role_index().get(ingredient_id, ()))


def role_feature_names() -> tuple[str, ...]:
    return ('biology:documented_role_present',) + tuple(
        f'biology:{role}:{target}' for role in ROLES for target in TARGETS)


def role_features(ingredient_id: str) -> tuple[float, ...]:
    """Binary roles with a known mask; absent observations never mean non-interaction."""
    facts = {(role['role'], role['target']) for role in roles_for_ingredient(ingredient_id)}
    return (float(bool(facts)),) + tuple(
        float((role, target) in facts) for role in ROLES for target in TARGETS)


def _exposure_state(texts: list[str], aliases: list[str]) -> str:
    """Conservative clause-level negation/uncertainty for whole-product exposure names."""
    states = set()
    for text in texts:
        value = text.casefold().replace('\u2019', "'").replace('\u2018', "'")
        for clause in re.split(r'[,;\n]|\bbut\b', value):
            for alias in aliases:
                for match in re.finditer(r'(?<![\w-])' + re.escape(alias) + r'(?![\w])', clause):
                    before, after = clause[:match.start()], clause[match.end():]
                    if re.search(r'\b(?:no|not|without|avoid|avoids|avoiding|never|denies)\b', before) or re.match(r'\s*-?free\b', after):
                        states.add('negated')
                    elif re.search(r'\b(?:maybe|possibly|possible|uncertain|unsure|might|unknown|sometimes|occasionally)\b', clause) or '?' in clause:
                        states.add('uncertain')
                    else:
                        states.add('consumed')
    if 'uncertain' in states or {'negated', 'consumed'} <= states:
        return 'uncertain'
    return next(iter(states)) if states else 'absent'


def _exposure_roles() -> dict[str, list[dict]]:
    exposures: dict[str, list[dict]] = {}
    for role in _snapshot()[0]['roles']:
        if role['entity_type'] != 'drug':
            exposures.setdefault(role['source_entity'], []).append(role)
    return exposures


def redacted_foods(foods: list[str]) -> list[str]:
    """Keep only canonical supported exposures and their context for private replay.

    This intentionally excludes unrelated free text. Merge it with the original
    food-rule redaction to preserve both bounded catalogs in exported reports.
    """
    result = []
    for roles in _exposure_roles().values():
        state = _exposure_state(foods, roles[0]['aliases'])
        if state == 'absent':
            continue
        prefix = {'consumed': '', 'negated': 'no ', 'uncertain': 'maybe '}[state]
        result.append(prefix + roles[0]['canonical_name'])
    return sorted(set(result))


def _hypothesis(modifier: dict, substrate: dict, owners: dict[str, list[str]],
                exposure_state: str | None = None) -> dict:
    first, second = modifier['canonical_name'], substrate['canonical_name']
    source_roles = [modifier, substrate]
    ingredient_ids = [role['ingredient_id'] for role in source_roles if role['ingredient_id']]
    conditions = list(dict.fromkeys(condition for role in source_roles for condition in role['conditions']))
    relationship = 'inhibits' if modifier['role'] == 'inhibitor' else 'induces'
    modifier_potency = '' if modifier['potency'] == 'clinical_table_category' else modifier['potency'].replace('_', ' ') + ' '
    substrate_potency = '' if substrate['potency'] == 'clinical_table_category' else substrate['potency'].replace('_', ' ') + ' '
    identity = {'modifier_role': modifier['role_id'], 'substrate_role': substrate['role_id'],
                'exposure_state': exposure_state}
    return {
        'hypothesis_id': _canonical_hash(identity)[:24],
        'status': 'mechanism_supported_possible',
        'entities': [first, second], 'ingredient_ids': ingredient_ids,
        'mention_ids': list(dict.fromkeys(mention for ident in ingredient_ids for mention in owners.get(ident, []))),
        'interaction_type': 'drug-drug' if modifier['entity_type'] == 'drug' else 'drug-food',
        'exposure_type': modifier['entity_type'], 'exposure_state': exposure_state,
        'path': [{'source': first, 'relation': relationship, 'target': modifier['target'],
                  'role_id': modifier['role_id']},
                 {'source': second, 'relation': 'substrate_of', 'target': substrate['target'],
                  'role_id': substrate['role_id']}],
        'mechanism': (f'{first} is listed as a {modifier_potency}'
                      f'{modifier["target"]} {modifier["role"]}; {second} is listed as a '
                      f'{substrate_potency}substrate of that pathway. '
                      'The shared pathway supports a possible interaction.'),
        'evidence': [{'role_id': role['role_id'], 'source': 'FDA clinical role table',
                      'url': role['source_url'], 'source_entity': role['source_entity'],
                      'excerpt': role['source_cell'], 'source_row_sha256': role['source_row_sha256'],
                      'footnote_ids': list(role['footnote_ids'])} for role in source_roles],
        'conditions': conditions,
        'limitations': [
            'This path is not a documented clinical pair outcome and assigns no severity or patient risk.',
            'Role potency is not pair severity; clinical effects depend on exposure and other pathways.',
            'Group labels and whole-product evidence are retained without unsupported expansion.',
        ],
    }


def mechanism_hypotheses(medications: list[Medication], foods: list[str] | None = None) -> list[dict]:
    """Derive sourced paths from resolved ingredients; candidates cannot create claims.

    Negated/uncertain supplement and food entries generate no biological hypothesis.
    The ordinary evidence/interaction engine remains authoritative for pair records.
    """
    known_ids = catalog()[1]
    owners: dict[str, list[str]] = {}
    for medication in medications:
        if not medication.normalized or medication.status not in ('exact', 'reviewed', 'resolved', 'confirmed'):
            continue
        for ident in medication.ingredient_ids:
            if ident in known_ids:
                owners.setdefault(ident, []).append(medication.mention_id)
    present = [role for ident in sorted(owners) for role in roles_for_ingredient(ident)]
    modifiers = [role for role in present if role['role'] in ('inhibitor', 'inducer')]
    substrates = [role for role in present if role['role'] == 'substrate']
    if foods:
        for exposure_roles in _exposure_roles().values():
            if _exposure_state(foods, exposure_roles[0]['aliases']) == 'consumed':
                modifiers.extend(role for role in exposure_roles if role['role'] in ('inhibitor', 'inducer'))
    results = {}
    for modifier in modifiers:
        for substrate in substrates:
            if modifier['target'] != substrate['target'] or modifier['ingredient_id'] == substrate['ingredient_id']:
                continue
            value = _hypothesis(modifier, substrate, owners,
                                'consumed' if modifier['entity_type'] != 'drug' else None)
            results[value['hypothesis_id']] = value
    return sorted(results.values(), key=lambda value: (value['entities'], value['path'][0]['target'], value['hypothesis_id']))
