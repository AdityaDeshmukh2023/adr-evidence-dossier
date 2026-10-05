"""Auditable pair and drug splits; ML labels stay out of input adjacency."""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import CLASSES, TASK
from .io import file_hash, object_hash, read_json, write_json

ALLOWED_NODE_FIELDS = {'ingredient_id', 'name', 'canonical_smiles', 'inchikey', 'fingerprint_bits', 'roles'}


def audit_predictive_features(dataset: dict) -> dict:
    """Whitelist feature schema to reject target labels or interaction prose in nodes."""
    for row in dataset['nodes']:
        if set(row) != ALLOWED_NODE_FIELDS:
            raise ValueError('Predictive feature leakage or unsupported node fields.')
        if (len(row['roles']) != len(dataset['role_feature_names'])
                or any(value not in (0, 1) for value in row['roles'])
                or any(not isinstance(bit, int) or not 0 <= bit < 2048 for bit in row['fingerprint_bits'])):
            raise ValueError('Invalid molecular or biology predictive features.')
    return {'passed': True, 'node_fields': sorted(ALLOWED_NODE_FIELDS),
            'uses_source_severity_as_feature': False, 'uses_interaction_text_as_feature': False,
            'source_severity_labels_are_supervision_only': True}


def pair_key(first: str, second: str) -> str:
    return '|'.join(sorted((first, second)))


def _partition(indices: list[int], labels: list[int], seed: int, fractions=(.7, .1, .1, .1)):
    groups = defaultdict(list)
    for index in indices:
        groups[labels[index]].append(index)
    parts = [[] for _ in fractions]
    rng = random.Random(seed)
    for label in sorted(groups):
        values = sorted(groups[label])
        rng.shuffle(values)
        cursor, cumulative = 0, 0.0
        for position, fraction in enumerate(fractions):
            cumulative += fraction
            end = len(values) if position == len(fractions) - 1 else int(len(values) * cumulative)
            parts[position].extend(values[cursor:end])
            cursor = end
    return [sorted(part) for part in parts]


def make_splits(pairs: list[dict], *, seed: int = 1729, chemical_keys: dict[str, str] | None = None) -> dict:
    labels = [CLASSES.index(row['severity']) for row in pairs]
    pool, validation, calibration, test = _partition(list(range(len(pairs))), labels, seed)
    message, supervision = _partition(pool, labels, seed + 1, (.5, .5))
    pair = {'message': message, 'supervision': supervision, 'validation': validation,
            'calibration': calibration, 'test': test}
    drugs = sorted({i for row in pairs for i in row['ingredient_ids']})
    rng = random.Random(seed)
    groups = defaultdict(list)
    for drug in drugs:
        if chemical_keys is not None and drug not in chemical_keys:
            raise ValueError('Every labeled drug needs a chemical key for cold splitting.')
        groups[chemical_keys[drug] if chemical_keys is not None else drug].append(drug)
    grouped_drugs = sorted(groups.values(), key=lambda values: tuple(values))
    rng.shuffle(grouped_drugs)
    ends = [int(len(drugs) * p) for p in (.7, .8, .9)] + [len(drugs)]
    partitions = {role: [] for role in ('train', 'validation', 'calibration', 'test')}
    position, count = 0, 0
    for group in grouped_drugs:
        while position < 3 and count >= ends[position]:
            position += 1
        partitions[('train', 'validation', 'calibration', 'test')[position]].extend(group)
        count += len(group)
    partitions = {role: sorted(values) for role, values in partitions.items()}
    assignments = {drug: role for role, values in partitions.items() for drug in values}
    cold = {'message': [], 'supervision': [], 'validation': [], 'calibration': [],
            'test_one_unseen': [], 'test_both_unseen': [], 'excluded_cross_role': []}
    cold_pool = []
    for index, row in enumerate(pairs):
        first, second = (assignments[ident] for ident in row['ingredient_ids'])
        roles = {first, second}
        if roles == {'train'}:
            cold_pool.append(index)
        elif roles <= {'train', 'validation'}:
            cold['validation'].append(index)
        elif roles <= {'train', 'calibration'}:
            cold['calibration'].append(index)
        elif roles == {'train', 'test'}:
            cold['test_one_unseen'].append(index)
        elif roles == {'test'}:
            cold['test_both_unseen'].append(index)
        else:
            cold['excluded_cross_role'].append(index)
    cold['message'], cold['supervision'] = _partition(cold_pool, labels, seed + 1, (.5, .5))
    splits = {'schema': 'ddi-splits-1', 'seed': seed, 'pair': pair, 'cold': cold,
              'cold_drugs': partitions, 'policy': '70/10/10/10; training message/supervision 50/50',
              'cold_grouping': 'exact_standardized_inchikey' if chemical_keys is not None else 'drug_identifier_only',
              'cold_chemical_keys': {ident: chemical_keys[ident] for ident in drugs} if chemical_keys is not None else {}}
    audit_splits(pairs, splits)
    return splits


def audit_splits(pairs: list[dict], splits: dict) -> dict:
    """Fail before training when reverse pairs, labels, or held-out drugs can leak."""
    keys = [pair_key(*row['ingredient_ids']) for row in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate or reversed source pair.')
    for row in pairs:
        if len(set(row['ingredient_ids'])) != 2 or row['severity'] not in CLASSES:
            raise ValueError('Invalid labeled pair.')
    summary = {}
    for protocol in ('pair', 'cold'):
        parts = splits[protocol]
        seen = set()
        for role, indices in parts.items():
            if len(indices) != len(set(indices)) or any(i < 0 or i >= len(pairs) for i in indices):
                raise ValueError(f'Invalid indices in {protocol}/{role}.')
            overlap = seen.intersection(indices)
            if overlap:
                raise ValueError(f'Pair-role leakage in {protocol}/{role}.')
            seen.update(indices)
        if seen != set(range(len(pairs))):
            raise ValueError(f'Unaccounted pairs in {protocol}.')
        summary[protocol] = {name: len(values) for name, values in parts.items()}
    drug_parts = splits['cold_drugs']
    if set(drug_parts) != {'train', 'validation', 'calibration', 'test'}:
        raise ValueError('Invalid cold-drug partition roles.')
    assignment = {}
    chemistry_roles = {}
    for role, ids in drug_parts.items():
        for ident in ids:
            if ident in assignment:
                raise ValueError('Drug-role leakage.')
            assignment[ident] = role
            chemical_key = splits.get('cold_chemical_keys', {}).get(ident)
            if chemical_key:
                if chemical_key in chemistry_roles and chemistry_roles[chemical_key] != role:
                    raise ValueError('Molecular-identity leakage across cold-drug partitions.')
                chemistry_roles[chemical_key] = role
    allowed = {'message': ({'train'},), 'supervision': ({'train'},),
               'validation': ({'validation'}, {'train', 'validation'}),
               'calibration': ({'calibration'}, {'train', 'calibration'}),
               'test_one_unseen': ({'train', 'test'},), 'test_both_unseen': ({'test'},)}
    for role, indices in splits['cold'].items():
        for index in indices:
            roles = {assignment[i] for i in pairs[index]['ingredient_ids']}
            if role == 'excluded_cross_role':
                if any(roles in options for options in allowed.values()):
                    raise ValueError('Incorrect excluded cold pair.')
            elif roles not in allowed[role]:
                raise ValueError(f'Held-out-drug leakage in {role}.')
    return {'passed': True, 'counts': summary, 'reversed_pair_leaks': 0,
            'supervision_in_adjacency': 0, 'heldout_drugs_in_cold_adjacency': 0}


def prepare_dataset(source_csv: str | Path, molecule_cache: str | Path,
                    output: str | Path, *, seed: int = 1729) -> dict:
    cache = read_json(molecule_cache)
    records = cache.get('records', {})
    nodes = {ident: row for ident, row in sorted(records.items()) if row.get('status') == 'mapped'
             and row.get('standardized_inchikey') == row.get('pubchem_inchikey')}
    if not nodes:
        raise ValueError('No molecular mappings available; run prepare_molecules.py first.')
    try:
        from adr_system.biology import role_feature_names, role_features, snapshot_manifest
        role_names = list(role_feature_names())
        role_manifest = snapshot_manifest()
        role_values = {ident: list(role_features(ident)) for ident in nodes}
    except ImportError:
        role_names, role_values, role_manifest = [], {ident: [] for ident in nodes}, {'status': 'unavailable'}
    feature_rows = [{'ingredient_id': ident, 'name': row['name'],
                     'canonical_smiles': row['canonical_smiles'],
                     'inchikey': row['standardized_inchikey'],
                     'fingerprint_bits': row['fingerprint_bits'], 'roles': role_values[ident]}
                    for ident, row in nodes.items()]
    seen, pairs, exclusions = set(), [], Counter()
    with Path(source_csv).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            ids = sorted((row['drug_a_id'], row['drug_b_id']))
            key = pair_key(*ids)
            if key in seen:
                raise ValueError(f'Duplicate/reversed pair in processed source: {key}')
            seen.add(key)
            if row['severity'] not in CLASSES:
                exclusions['unknown_source_severity'] += 1
                continue
            if not all(ident in nodes for ident in ids):
                exclusions['missing_or_excluded_molecule'] += 1
                continue
            pairs.append({'ingredient_ids': ids, 'severity': row['severity'], 'source_record_id': key})
    if not pairs:
        raise ValueError('No eligible labeled pairs after molecular filtering.')
    dataset = {'schema': 'ddi-dataset-1', 'task': TASK, 'classes': list(CLASSES),
               'fingerprint': cache['fingerprint'], 'nodes': feature_rows, 'pairs': pairs,
               'role_feature_names': role_names, 'biology_manifest': role_manifest,
               'source_csv_sha256': file_hash(source_csv), 'molecule_cache_sha256': file_hash(molecule_cache),
               'exclusions': dict(exclusions), 'class_counts': dict(Counter(p['severity'] for p in pairs)),
               'scope': 'Predict source severity conditional on documented interactions; no safe-negative labels.'}
    dataset['content_sha256'] = object_hash(dataset)
    feature_audit = audit_predictive_features(dataset)
    splits = make_splits(pairs, seed=seed, chemical_keys={row['ingredient_id']: row['inchikey'] for row in feature_rows})
    splits['dataset_sha256'] = dataset['content_sha256']
    audit = audit_splits(pairs, splits)
    output = Path(output)
    write_json(output / 'dataset.json', dataset)
    write_json(output / 'splits.json', splits)
    write_json(output / 'split_audit.json', audit)
    write_json(output / 'feature_audit.json', feature_audit)
    write_json(output / 'coverage.json', {'catalog_size': cache.get('catalog_size'),
                                         'attempted_mappings': len(records), 'eligible_molecules': len(nodes),
                                         'identity_mismatch_exclusions': sum(row.get('status') == 'mapped' and
                                           row.get('standardized_inchikey') != row.get('pubchem_inchikey')
                                           for row in records.values()),
                                         'documented_role_molecules': sum(bool(row['roles'] and row['roles'][0])
                                                                          for row in feature_rows),
                                         'eligible_pairs': len(pairs), 'class_counts': dataset['class_counts'],
                                         'pair_exclusions': dict(exclusions), 'cache_status': cache.get('status')})
    return {'nodes': len(feature_rows), 'eligible_pairs': len(pairs), 'class_counts': dataset['class_counts'],
            'exclusions': dict(exclusions), 'role_features': len(role_names), 'audit': audit}
