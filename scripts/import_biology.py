"""Import a bounded, source-hashed FDA clinical enzyme/transporter role snapshot.

The source is a role table, not a set of pairwise clinical interaction labels.
Network refresh is explicit; ordinary screening always reads the frozen JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_system.terminology import exact

SOURCE_URL = ('https://www.fda.gov/drugs/drug-interactions-labeling/'
              'healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems')
SOURCE_TITLE = 'FDA Examples of Drugs that Interact with CYP Enzymes and Transporter Systems'
IMPORTER_VERSION = '1.0'
OUTPUT = ROOT / 'data' / 'biological_roles.json'
SELECTED_DRUGS = (
    'amiodarone', 'atorvastatin', 'buspirone', 'caffeine', 'carbamazepine',
    'cimetidine', 'ciprofloxacin', 'clarithromycin', 'clopidogrel', 'colchicine',
    'cyclosporine', 'dabigatran etexilate', 'dextromethorphan', 'digoxin',
    'diltiazem', 'fluconazole', 'fluoxetine', 'fluvoxamine', 'gemfibrozil',
    'itraconazole', 'ketoconazole', 'lansoprazole', 'lovastatin', 'metformin',
    'midazolam', 'omeprazole', 'paroxetine', 'repaglinide', 'rifampin',
    'rosuvastatin', 'simvastatin', 'tacrolimus', 'terbinafine', 'tizanidine',
    'verapamil', 'warfarin',
)
SELECTED_EXPOSURES = {
    'grapefruit juice': {'kind': 'food', 'aliases': ['grapefruit juice']},
    "st. john's wort": {'kind': 'supplement', 'aliases': [
        "st. john's wort", "st john's wort", 'st johns wort', 'st. johns wort']},
    'curcumin': {'kind': 'supplement', 'aliases': ['curcumin']},
    'diosmin': {'kind': 'supplement', 'aliases': ['diosmin']},
}
COLUMNS = (
    ('inhibitor', 'strong', 'CYP'), ('inhibitor', 'moderate', 'CYP'),
    ('inhibitor', 'weak', 'CYP'), ('inducer', 'strong', 'CYP'),
    ('inducer', 'moderate', 'CYP'), ('inducer', 'weak', 'CYP'),
    ('substrate', 'sensitive', 'CYP'), ('substrate', 'moderately_sensitive', 'CYP'),
    ('inhibitor', 'clinical_table_category', 'transporter'),
    ('substrate', 'clinical_table_category', 'transporter'),
)
TARGET_PATTERN = re.compile(
    r'(?<![A-Za-z0-9])(?:CYP)?(1A2|2B6|2C8|2C9|2C19|2D6|3A)(?![A-Za-z0-9])')
TRANSPORTER_PATTERN = re.compile(
    r'(?<![A-Za-z0-9])(?:P-gp|BCRP|OATP1B1|OATP1B3|OATP1B|OAT1|OAT3|OCT2|MATE1|MATE2-K)(?![A-Za-z0-9])')


def normalized(value: str) -> str:
    return ' '.join(value.replace('\u2019', "'").replace('\u2018', "'").split())


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


class FDATableParser(HTMLParser):
    """Retain row text and superscript footnotes without mixing their numbers."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.paragraphs = []
        self._row = None
        self._cell = None
        self._paragraph = None
        self._sup = False

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._row = []
        elif tag in ('td', 'th') and self._row is not None:
            self._cell = {'text': '', 'footnotes': []}
        elif tag == 'p':
            self._paragraph = {'text': '', 'footnotes': []}
        elif tag == 'sup':
            self._sup = True

    def handle_endtag(self, tag):
        if tag == 'sup':
            self._sup = False
        elif tag in ('td', 'th') and self._cell is not None:
            self._cell['text'] = normalized(self._cell['text'])
            self._row.append(self._cell)
            self._cell = None
        elif tag == 'tr' and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == 'p' and self._paragraph is not None:
            self._paragraph['text'] = normalized(self._paragraph['text'])
            self.paragraphs.append(self._paragraph)
            self._paragraph = None

    def handle_data(self, value):
        for item in (self._cell, self._paragraph):
            if item is None:
                continue
            if self._sup:
                item['footnotes'].extend(re.findall(r'\d+', value))
            else:
                item['text'] += value


def build_snapshot(raw_html: bytes, retrieved_at: str) -> dict:
    parser = FDATableParser()
    parser.feed(raw_html.decode('utf-8'))
    if not any(len(row) == 11 and row[0]['text'] == 'Drug or Other Substance' for row in parser.rows):
        raise ValueError('FDA role table header not found; snapshot was not written.')
    notes = {}
    for paragraph in parser.paragraphs:
        if len(paragraph['footnotes']) == 1:
            note = paragraph['footnotes'][0]
            if note in notes and notes[note] != paragraph['text']:
                continue
            notes[note] = paragraph['text']
    wanted = set(SELECTED_DRUGS) | set(SELECTED_EXPOSURES)
    selected_rows = []
    roles = []
    mapped, unmapped = {}, []
    for source_row in parser.rows:
        if len(source_row) != 11:
            continue
        name = normalized(source_row[0]['text']).casefold()
        if name not in wanted:
            continue
        kind = SELECTED_EXPOSURES.get(name, {}).get('kind', 'drug')
        options = exact(name) if kind == 'drug' else ()
        ident = options[0].ingredient_ids[0] if len(options) == 1 and len(options[0].ingredient_ids) == 1 else None
        if kind == 'drug':
            if ident:
                mapped[name] = ident
            else:
                unmapped.append(name)
        row_record = {'source_entity': name, 'cells': source_row,
                      'row_sha256': canonical_hash(source_row)}
        selected_rows.append(row_record)
        row_notes = set(source_row[0]['footnotes'])
        for cell, (role, potency, family) in zip(source_row[1:], COLUMNS):
            if not cell['text']:
                continue
            targets = ([f'CYP{match}' for match in TARGET_PATTERN.findall(cell['text'])]
                       if family == 'CYP' else TRANSPORTER_PATTERN.findall(cell['text']))
            if not targets:
                raise ValueError(f'Cannot interpret sourced role: {name}: {cell["text"]}')
            note_ids = sorted(row_notes | set(cell['footnotes']), key=int)
            if any(note not in notes for note in note_ids):
                raise ValueError(f'Missing source footnote for {name}: {note_ids}')
            for target in dict.fromkeys(targets):
                source_fact = {'source_entity': name, 'role': role, 'target': target,
                               'potency': potency, 'source_cell': cell['text'],
                               'source_row_sha256': row_record['row_sha256']}
                record = {**source_fact, 'role_id': canonical_hash(source_fact)[:24],
                          'entity_type': kind, 'ingredient_id': ident,
                          'canonical_name': options[0].names[0] if ident else name,
                          'target_family': family,
                          'evidence_type': 'FDA_clinical_role_table',
                          'footnote_ids': note_ids,
                          'conditions': [notes[note] for note in note_ids],
                          'source_url': SOURCE_URL,
                          'aliases': SELECTED_EXPOSURES.get(name, {}).get('aliases', [])}
                roles.append(record)
    missing = sorted(wanted - {row['source_entity'] for row in selected_rows})
    if missing:
        raise ValueError(f'Selected FDA rows missing: {missing}; inspect source changes before refreshing.')
    roles.sort(key=lambda item: (item['source_entity'], item['target'], item['role'], item['potency']))
    selected_rows.sort(key=lambda item: (item['source_entity'], item['row_sha256']))
    snapshot = {
        'schema_version': '1.0', 'version': f'fda-clinical-roles-{retrieved_at[:10]}',
        'source': {'title': SOURCE_TITLE, 'publisher': 'U.S. Food and Drug Administration',
                   'url': SOURCE_URL, 'section': 'Table 1: clinical CYP and transporter roles',
                   'retrieved_at': retrieved_at, 'raw_document_sha256': hashlib.sha256(raw_html).hexdigest(),
                   'selected_rows_sha256': canonical_hash(selected_rows),
                   'text_encoding': 'utf-8'},
        'import': {'importer': 'scripts/import_biology.py', 'importer_version': IMPORTER_VERSION,
                   'selected_drugs': list(SELECTED_DRUGS),
                   'selected_other_substances': sorted(SELECTED_EXPOSURES),
                   'ingredient_mappings': dict(sorted(mapped.items())),
                   'unmapped_drugs': sorted(set(unmapped)),
                   'normalization': 'whitespace and curly apostrophe normalization; exact ingredient lookup only'},
        'limitations': [
            'Bounded, non-exhaustive clinical role table; absent roles are unknown, not negative evidence.',
            'A shared pathway supports a mechanistic possibility, not a documented clinical pair outcome.',
            'Role potency describes source categories, not pair interaction severity or individual risk.',
            'CYP3A is retained as a group and is not expanded into CYP3A4 or CYP3A5.',
            'Dose, route, enantiomer and preparation footnotes must accompany derived paths.',
            'Whole-product supplement roles are not generalized to constituent-containing herbs or foods.',
        ],
        'selected_rows': selected_rows, 'roles': roles,
    }
    snapshot['roles_sha256'] = canonical_hash(roles)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--refresh', action='store_true', help='Download the public FDA role table.')
    mode.add_argument('--from-html', type=Path, help='Import an already downloaded FDA HTML file.')
    mode.add_argument('--check', action='store_true', help='Validate the frozen snapshot without network access.')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--retrieved-at', help='UTC ISO timestamp; defaults to actual fetch/import time.')
    args = parser.parse_args()
    if args.check:
        from adr_system.biology import validate_snapshot
        value = json.loads(args.output.read_text(encoding='utf-8'))
        validate_snapshot(value)
    else:
        if args.refresh:
            import requests
            response = requests.get(SOURCE_URL, timeout=40, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36'})
            response.raise_for_status()
            raw_html = response.content
        else:
            raw_html = args.from_html.read_bytes()
        retrieved_at = args.retrieved_at or datetime.now(timezone.utc).isoformat()
        value = build_snapshot(raw_html, retrieved_at)
        # A malformed or drifting source must never replace the previous snapshot.
        from adr_system.biology import validate_snapshot
        validate_snapshot(value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'snapshot': str(args.output), 'version': value['version'],
                      'roles': len(value['roles']), 'mapped_drugs': len(value['import']['ingredient_mappings']),
                      'unmapped_drugs': value['import']['unmapped_drugs']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
