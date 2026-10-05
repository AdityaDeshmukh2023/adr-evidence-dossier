"""Offline identifier resolution. Interaction outcomes never influence candidates."""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

from .data import ddinter_drug_index, load_json
from .models import Candidate


def clean_name(value: str) -> str:
    return ' '.join(value.casefold().strip().split())


@lru_cache(maxsize=1)
def catalog() -> tuple[dict[str, tuple[Candidate, ...]], dict[str, str]]:
    by_name: dict[str, list[Candidate]] = {}
    id_names: dict[str, str] = {}
    for name, record in ddinter_drug_index().items():
        ident = record['id']
        id_names[ident] = name
        by_name.setdefault(name, []).append(Candidate((ident,), (name,), 1.0, 'exact'))
    from .data import load_interactions, load_food_rules
    demo, _ = load_interactions()
    names = {n for pair in demo['interactions'] for n in pair.split('|')}
    names |= {r['drug'] for r in load_food_rules()[0]['rules']}
    names |= {'acetaminophen', 'metformin', 'amoxicillin'}
    names.discard('aspirin')
    names.add('acetylsalicylic acid')
    for name in sorted(names):
        if name not in by_name:
            ident = f'local:{name}'
            id_names[ident] = name
            by_name[name] = [Candidate((ident,), (name,), 1.0, 'exact')]
    config, _ = load_json('terminology.json')
    for alias, ingredients in {**config['aliases'], **config['combination_products']}.items():
        options = [by_name.get(clean_name(n), []) for n in ingredients]
        if not options or any(len(items) != 1 for items in options):
            raise ValueError(f'Terminology alias has missing or ambiguous ingredients: {alias}')
        ids = tuple(sorted({i for items in options for i in items[0].ingredient_ids}))
        candidate = Candidate(ids, tuple(id_names[i] for i in ids), 1.0, 'alias')
        by_name.setdefault(clean_name(alias), []).append(candidate)
    return ({name: tuple({c.ingredient_ids: c for c in items}.values())
             for name, items in by_name.items()}, id_names)


def exact(value: str) -> tuple[Candidate, ...]:
    return catalog()[0].get(clean_name(value), ())


@lru_cache(maxsize=1024)
def suggest(value: str, limit: int = 5) -> tuple[Candidate, ...]:
    value = clean_name(value)
    if exact(value):
        return exact(value)
    if len(value) < 3 or len(value) > 100:
        return ()
    terms, _ = catalog()
    variants = {value, value.translate(str.maketrans({'0': 'o', '1': 'l', '5': 's'}))}
    ranked: dict[tuple[str, ...], Candidate] = {}
    for term, options in terms.items():
        if abs(len(term) - len(value)) > max(4, len(value) // 2):
            continue
        score = max(difflib.SequenceMatcher(None, v, term).ratio() for v in variants)
        if score < 0.60:
            continue
        for option in options:
            if option.ingredient_ids not in ranked or ranked[option.ingredient_ids].score < score:
                ranked[option.ingredient_ids] = Candidate(option.ingredient_ids, option.names, round(score, 5))
    return tuple(sorted(ranked.values(), key=lambda c: (-c.score, c.ingredient_ids))[:limit])


@lru_cache(maxsize=1)
def name_pattern() -> re.Pattern:
    alternatives = '|'.join(re.escape(n) for n in sorted(catalog()[0], key=lambda n: (-len(n), n)))
    return re.compile(r'(?<![\w-])(?:' + alternatives + r')(?![\w-])', re.IGNORECASE)
