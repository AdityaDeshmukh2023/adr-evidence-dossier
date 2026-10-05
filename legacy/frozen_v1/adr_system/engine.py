from __future__ import annotations

import uuid

import itertools

from .data import find_ddinter_pair, load_ddinter_lookup, load_food_rules, load_interactions
from .models import AnalysisResult, Evidence, InteractionAlert, Medication
from .normalization import normalise_foods, parse_medication_lines


def _evidence(record: dict) -> list[Evidence]:
    return [Evidence(title=item["title"], url=item["url"], excerpt=item["excerpt"],
                     source=item["source"], retrieved_at=item["retrieved_at"])
            for item in record["evidence"]]


def analyze_medications(medication_text: str, food_text: str = "") -> AnalysisResult:
    medications = parse_medication_lines(medication_text)
    foods = normalise_foods(food_text)
    ddinter, dd_hash = load_interactions()
    full_lookup, full_hash = load_ddinter_lookup()
    food_rules, food_hash = load_food_rules()
    result = AnalysisResult(
        run_id=str(uuid.uuid4()), medications=medications, foods=foods,
        unsupported_medications=[m.name for m in medications if not m.normalized],
        source_status={"curated_ddi": f"snapshot {ddinter['version']} ({dd_hash[:12]})",
                       "curated_food_rules": f"snapshot {food_rules['version']} ({food_hash[:12]})"},
    )
    if full_lookup:
        result.source_status["ddinter"] = f"snapshot {full_lookup['version']} ({full_hash[:12]})"
    canonicals = [m.canonical_name for m in medications if m.canonical_name]
    for first, second in itertools.combinations(sorted(set(canonicals)), 2):
        record = find_ddinter_pair(first, second)
        if record:
            result.alerts.append(InteractionAlert(
                entities=[record["drug_a_name"], record["drug_b_name"]], interaction_type="drug-drug", severity=record["severity"],
                mechanism="The source snapshot provides a severity level but no mechanism/management statement.",
                management="Consult current labeling and a qualified clinician or pharmacist; do not change treatment based on this tool.",
                evidence=[Evidence(title="DDInter processed interaction snapshot", url="https://ddinter.scbdd.com/download/", excerpt=f"DDInter source severity: {record['severity_source']}; source categories: {record['source_categories']}.", source="DDInter", retrieved_at=full_lookup["version"])],
                data_version=full_lookup["version"], confidence=1.0, source_record_id=f"{record['drug_a_id']}|{record['drug_b_id']}",
            ))
            continue
        key = "|".join((first, second))
        record = ddinter["interactions"].get(key)
        if record:
            result.alerts.append(InteractionAlert(
                entities=[first, second], interaction_type="drug-drug", severity=record["severity"],
                mechanism=record["mechanism"], management=record["management"], evidence=_evidence(record),
                data_version=ddinter["version"], confidence=record["confidence"], source_record_id=record["id"],
            ))
    for drug in sorted(set(canonicals)):
        for food in foods:
            for rule in food_rules["rules"]:
                if drug == rule["drug"] and any(term in food for term in rule["food_terms"]):
                    result.alerts.append(InteractionAlert(
                        entities=[drug, food], interaction_type="drug-food", severity=rule["severity"],
                        mechanism=rule["mechanism"], management=rule["management"], evidence=_evidence(rule),
                        data_version=food_rules["version"], confidence=rule["confidence"], source_record_id=rule["id"],
                    ))
    if not result.alerts:
        result.warnings.append("No known interaction was found in the configured knowledge base. This does not mean the combination is safe.")
    if result.unsupported_medications:
        result.warnings.append("Some medication names were not recognized. Edit the spelling or consult a clinician/pharmacist.")
    return result


def severity_rank(value: str) -> int:
    return {"high": 3, "moderate": 2, "low": 1}.get(value, 0)
