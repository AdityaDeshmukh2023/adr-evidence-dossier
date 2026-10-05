from __future__ import annotations

import itertools
import os
from copy import deepcopy
import re
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from .data import load_food_rules
from .knowledge import evidence_from, pair_alert, pair_state
from .models import AnalysisResult, Candidate, InteractionAlert, Medication, ReviewDecision, ReviewItem
from .normalization import MAX_MEDICATIONS, normalise_foods, parse_medication_lines
from .provenance import manifest
from .terminology import catalog, exact


def severity_rank(value: str) -> int:
    return {'high': 3, 'moderate': 2, 'low': 1}.get(value, 0)


def _options(med: Medication) -> tuple[Candidate, ...]:
    if med.ingredient_ids:
        return (Candidate(med.ingredient_ids, med.ingredient_names, 1, med.status),)
    return med.candidates[:5]


def review_priorities(medications: list[Medication]) -> list[ReviewItem]:
    queue = []
    for med in medications:
        if med.status in ('reviewed', 'excluded'):
            continue
        options = _options(med)
        high, other, coverage = 0, 0, 0
        previews = []
        for partner in medications:
            if partner.mention_id == med.mention_id or partner.status == 'excluded':
                continue
            signatures = set()
            if not med.normalized:
                signatures.add((('', '', 'unassessed'),))
            for choice in options:
                for partner_choice in _options(partner):
                    findings = tuple(sorted((a, b, pair_state(a, b))
                                           for a in choice.ingredient_ids for b in partner_choice.ingredient_ids))
                    # Compare findings with identities as well as severity: two
                    # different high-severity pairs still require identity review.
                    signatures.add(findings)
                    if not med.normalized:
                        previews.append({'candidate': list(choice.names), 'partner_mention': partner.mention_id,
                                         'partner_candidate': list(partner_choice.names),
                                         'states': [state for _, _, state in findings],
                                         'status': 'candidate-dependent; not a confirmed alert'})
            if len(signatures) > 1:
                states = {s for sig in signatures for _, _, s in sig}
                high += int('high' in states)
                other += int(bool(states & {'moderate', 'low'}))
                coverage += int(bool(states & {'no_record', 'unknown', 'unassessed'}))
        required = not options
        scores = sorted((c.score for c in med.candidates), reverse=True)
        ambiguity = 1 - (scores[0] - scores[1]) if len(scores) > 1 else (1 - scores[0] if scores else 1)
        if med.ocr_score is not None:
            ambiguity = max(ambiguity, 1 - med.ocr_score)
        reason = ('No usable candidate: manual identification is required.' if required else
                  'Resolving this reading can change high-severity source findings.' if high else
                  'Candidate identities change other source findings.' if other else
                  'Candidate identities change source coverage or unknown-severity findings.' if coverage else
                  'Verify the reading; no candidate-dependent interaction change was found in this snapshot.')
        # Exact matches stay reviewable but never outrank an unresolved item solely
        # because its partner is ambiguous.
        if med.normalized:
            high = other = coverage = 0
            reason = 'Exact terminology match; verify against the source prescription.'
        queue.append(ReviewItem(med.mention_id, required, high, other, coverage,
                                round(ambiguity, 5), reason, previews))
    order = {m.mention_id: i for i, m in enumerate(medications)}
    return sorted(queue, key=lambda q: (not q.required, -q.high_changes, -q.other_changes,
                                       -q.coverage_changes, -q.ambiguity, order[q.mention_id]))


def _food_state(food: str, terms: list[str]) -> str:
    states = set()
    for term in sorted(terms, key=len, reverse=True):
        for match in re.finditer(r'(?<!\w)' + re.escape(term) + r'(?!\w)', food):
            prefix = re.split(r'\bbut\b', food[:match.start()])[-1]
            suffix = food[match.end():]
            if re.search(r'\b(?:no|not|without|avoid|avoiding)\b', prefix) or re.match(r'[- ]free\b', suffix):
                states.add('negated')
            elif re.search(r'\b(?:maybe|occasionally|unsure|possibly)\b', prefix):
                states.add('uncertain')
            else:
                states.add('consumed')
    if 'uncertain' in states or {'negated', 'consumed'} <= states:
        return 'uncertain'
    return next(iter(states)) if states else 'no_rule'


def analyze_structured(medications: list[Medication], foods: list[str] | None = None,
                       *, history: list[ReviewDecision] | None = None,
                       run_id: str | None = None, include_review: bool = True,
                       include_hybrid: bool = True, enable_research_model: bool = False,
                       model_dir: str | None = None) -> AnalysisResult:
    if len(medications) > MAX_MEDICATIONS:
        raise ValueError(f'Use at most {MAX_MEDICATIONS} entries.')
    medications = [replace(m, mention_id=m.mention_id or f'm{i + 1}') for i, m in enumerate(medications)]
    if len({m.mention_id for m in medications}) != len(medications):
        raise ValueError('Medication mention IDs must be unique.')
    names = catalog()[1]
    for med in medications:
        if any(i not in names for i in med.ingredient_ids):
            raise ValueError('Unknown ingredient identifier.')
        if bool(med.ingredient_ids) != med.normalized or (med.status == 'excluded' and med.ingredient_ids):
            raise ValueError('Medication resolution status and ingredient IDs disagree.')
        if len(med.candidates) > 5 or any(not 0 <= c.score <= 1 or any(i not in names for i in c.ingredient_ids) for c in med.candidates):
            raise ValueError('Invalid medication candidate set.')
    medications = [replace(m, ingredient_names=tuple(names[i] for i in m.ingredient_ids),
                           canonical_name=names[m.ingredient_ids[0]] if m.ingredient_ids else None) for m in medications]
    foods = list(dict.fromkeys(foods or []))
    result = AnalysisResult(run_id or str(uuid.uuid4()), medications, foods,
                            review_history=list(history or []), manifest=deepcopy(manifest()))
    result.source_status = {name: digest or 'absent' for name, digest in result.manifest['artifacts'].items()}
    active = [m for m in medications if m.status != 'excluded']
    unresolved = [m for m in active if not m.normalized or not m.ingredient_ids]
    result.unsupported_medications = [m.name for m in unresolved]
    result.completeness = 'empty' if not active else ('incomplete' if unresolved else 'complete_for_entered_medicines')
    owners: dict[str, list[str]] = {}
    for med in active:
        for ident in med.ingredient_ids:
            owners.setdefault(ident, []).append(med.mention_id)
    for first, second in itertools.combinations(sorted(owners), 2):
        alert = pair_alert(first, second)
        result.pair_assessments.append({'ingredient_ids': [first, second],
                                        'state': alert.severity if alert else 'no_record'})
        if alert:
            alert.mention_ids = sorted(set(owners[first] + owners[second]))
            result.alerts.append(alert)
    for ident, mentions in owners.items():
        if len(mentions) > 1:
            result.warnings.append(f'Repeated ingredient {names[ident]} in entries {", ".join(mentions)}; confirm intentional duplication.')
    for first, second in itertools.combinations(active, 2):
        if not first.ingredient_ids or not second.ingredient_ids:
            result.pair_assessments.append({'mention_ids': [first.mention_id, second.mention_id], 'state': 'unassessed'})
    rules, _ = load_food_rules()
    seen_food_rules = set()
    for food in foods:
        matched = False
        for rule in rules['rules']:
            match = exact(rule['drug'])
            if not match or not set(match[0].ingredient_ids) & owners.keys():
                continue
            state = _food_state(food, rule['food_terms'])
            if state == 'no_rule':
                continue
            matched = True
            result.food_assessments.append({'food': food, 'rule_id': rule['id'], 'state': state})
            if state == 'uncertain':
                result.warnings.append('Food consumption is uncertain; confirm the dietary entry before assessment.')
            if state != 'consumed':
                continue
            if rule['id'] in seen_food_rules:
                continue
            seen_food_rules.add(rule['id'])
            ids = list(match[0].ingredient_ids)
            result.alerts.append(InteractionAlert([rule['drug'], rule['food_terms'][0]], 'drug-food', rule['severity'],
                rule['mechanism'], rule['management'], evidence_from(rule), rules['version'], None, rule['id'],
                ingredient_ids=ids, mention_ids=sorted({m for i in ids for m in owners.get(i, [])})))
        if not matched:
            result.food_assessments.append({'food': food, 'state': 'no_rule'})
    if not active:
        result.warnings.append('No medication entries were available to assess.')
    if not result.alerts:
        result.warnings.append('No known interaction was found in the configured knowledge base. This does not mean the combination is safe.')
    if unresolved:
        result.warnings.append('Assessment is incomplete: unresolved entries and their pairs have not been assessed.')
    if any(a.severity == 'unknown' for a in result.alerts):
        result.warnings.append('Some source records have unknown severity. Unknown does not mean low risk.')
    if include_review:
        result.review_queue = review_priorities(medications)
    result.manifest['hybrid_config'] = {'include_hybrid': include_hybrid,
                                       'enable_research_model': enable_research_model}
    if include_hybrid:
        from .uncertainty import candidate_assessments
        result.candidate_assessments = candidate_assessments(medications)
        result.hybrid_status['candidate_scope'] = 'Exact pair joins within retained candidates; identity coverage is not calibrated.'
        try:
            from .biology import mechanism_hypotheses, snapshot_manifest
            result.mechanism_hypotheses = mechanism_hypotheses(medications, foods)
            result.hybrid_status['biology'] = {'status': 'available', **snapshot_manifest()}
        except ImportError:
            result.hybrid_status['biology'] = {'status': 'unavailable'}
        except (OSError, ValueError) as error:
            result.mechanism_hypotheses = []
            result.hybrid_status['biology'] = {'status': 'incompatible', 'reason': str(error)[:200]}
    result.hybrid_status['model'] = {'status': 'disabled'}
    if enable_research_model:
        from .data import ROOT
        directory = model_dir or os.getenv('HYBRID_MODEL_DIR') or str(ROOT / 'artifacts/ml/deployment')
        try:
            from .ml.inference import predict_unknown, bundle_manifest
            result.hybrid_status['model'] = bundle_manifest(directory)
            result.model_predictions = predict_unknown(result, directory)
        except ImportError:
            result.hybrid_status['model'] = {'status': 'unavailable', 'reason': 'Optional ML profile is not installed.'}
        except (OSError, ValueError, RuntimeError) as error:
            result.hybrid_status['model'] = {'status': 'incompatible', 'reason': str(error)[:200]}
        result.manifest['model_artifacts'] = result.hybrid_status['model']
    return result


def analyze_medications(medication_text: str, food_text: str = '', **options) -> AnalysisResult:
    return analyze_structured(parse_medication_lines(medication_text), normalise_foods(food_text), **options)


def apply_review(result: AnalysisResult, mention_id: str, ingredient_ids: tuple[str, ...],
                 *, action: str = 'resolve') -> AnalysisResult:
    if action not in ('resolve', 'exclude_non_medication'):
        raise ValueError('Invalid review action.')
    selected = next((m for m in result.medications if m.mention_id == mention_id), None)
    if selected is None:
        raise ValueError('Unknown medication mention.')
    ids = tuple(sorted(set(ingredient_ids)))
    names = catalog()[1]
    if action == 'resolve' and (not ids or any(i not in names for i in ids)):
        raise ValueError('Choose valid ingredient IDs or enter a recognized medicine.')
    if action == 'exclude_non_medication':
        ids = ()
    updated = replace(selected, ingredient_ids=ids, ingredient_names=tuple(names[i] for i in ids),
                      canonical_name=names[ids[0]] if ids else None, normalized=bool(ids),
                      status='reviewed' if ids else 'excluded', unparsed_text='')
    medications = [updated if m.mention_id == mention_id else m for m in result.medications]
    next_result = analyze_structured(medications, result.foods, run_id=result.run_id, history=result.review_history,
                                    **result.manifest.get('hybrid_config', {}))
    decision = ReviewDecision(mention_id, ids, selected.ingredient_ids, datetime.now(timezone.utc).isoformat(), action,
                              tuple(sorted(a.source_record_id for a in result.alerts)),
                              tuple(sorted(a.source_record_id for a in next_result.alerts)))
    next_result.review_history.append(decision)
    return next_result
