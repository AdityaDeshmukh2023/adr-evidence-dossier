from __future__ import annotations

import re
from dataclasses import replace

from .models import Medication
from .terminology import clean_name, exact, name_pattern, suggest

MAX_INPUT_CHARS = 10000
MAX_MEDICATIONS = 30
DOSE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:mcg|mg|ml|g|iu|units?|%)(?:\s*/\s*(?:\d+(?:\.\d+)?\s*)?(?:ml|g))?(?!\w)', re.I)
FREQUENCY = re.compile(r'\b(?:once daily|twice daily|daily|nightly|weekly|bid|tid|qid|od|bd|tds|prn|at night|every \d+ hours?|\d-\d-\d)\b', re.I)
ROUTE = re.compile(r'\b(?:oral|orally|topical|intravenous|intramuscular|po|iv|im)\b', re.I)
FORM = re.compile(r'\b(?:tab(?:let)?s?|cap(?:sule)?s?|syrup|take)\b\.?\s*', re.I)


def _attributes(text: str):
    dose, freq, route = DOSE.search(text), FREQUENCY.search(text), ROUTE.search(text)
    remainder = FORM.sub('', ROUTE.sub('', FREQUENCY.sub('', DOSE.sub('', text))))
    remainder = re.sub(r'^\s*\d+[.)]\s*', '', remainder)
    remainder = remainder.strip(' \t,+;:/.-')
    return dose.group() if dose else None, freq.group() if freq else None, route.group() if route else None, remainder


def normalise_medication(value: str) -> Medication:
    if exact(value):
        strength, frequency, route, name = None, None, None, value
    else:
        matches = list(name_pattern().finditer(value))
        if len(matches) == 1:
            match = matches[0]
            strength, frequency, route, remainder = _attributes(value[:match.start()] + value[match.end():])
            name = match.group() if not remainder else _attributes(value)[3]
        else:
            strength, frequency, route, name = _attributes(value)
    choices = exact(name) or suggest(name)
    resolved = len(exact(name)) == 1
    selected = choices[0] if resolved else None
    return Medication(name=value.strip(), canonical_name=selected.names[0] if selected else None,
                      strength=strength, frequency=frequency, route=route, normalized=resolved,
                      ingredient_ids=selected.ingredient_ids if selected else (),
                      ingredient_names=selected.names if selected else (), candidates=choices,
                      status='exact' if resolved else ('ambiguous' if choices else 'unresolved'),
                      unparsed_text='' if resolved else name)


def parse_medication_lines(text: str) -> list[Medication]:
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f'Medication input exceeds {MAX_INPUT_CHARS:,} characters.')
    mentions = []
    for line in re.finditer(r'[^\n,;]+', text):
        raw = line.group()
        if not raw.strip():
            continue
        matches = list(name_pattern().finditer(raw))
        segments = []
        if not matches:
            segments = [(0, len(raw))]
        else:
            prefix = raw[:matches[0].start()]
            if _attributes(prefix)[3]:
                segments.append((0, matches[0].start()))
            for i, match in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
                tail = raw[match.end():end]
                if _attributes(tail)[3]:
                    attr_end = match.end()
                    attr_match = re.match(r'\s*\d+(?:\.\d+)?\s*(?:mcg|mg|ml|g)\b', tail, re.I)
                    if attr_match:
                        attr_end += attr_match.end()
                    segments.append((match.start(), attr_end))
                    segments.append((attr_end, end))
                else:
                    start = 0 if i == 0 and not _attributes(prefix)[3] else match.start()
                    segments.append((start, end))
        for start, end in segments:
            piece = raw[start:end].strip(' \t+/')
            if not piece:
                continue
            item = normalise_medication(piece)
            mentions.append(replace(item, mention_id=f'm{len(mentions) + 1}',
                                    source_span=(line.start() + start, line.start() + end)))
    if len(mentions) > MAX_MEDICATIONS:
        raise ValueError(f'Use at most {MAX_MEDICATIONS} medication entries per analysis.')
    return mentions


def normalise_foods(text: str) -> list[str]:
    if len(text) > 2000:
        raise ValueError('Food input exceeds 2,000 characters.')
    return list(dict.fromkeys(clean_name(x) for x in re.split(r'[\n,;]+', text) if x.strip()))
