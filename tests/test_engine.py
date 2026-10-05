from adr_system.engine import analyze_medications
from adr_system.normalization import normalise_medication
from adr_system.report import render_html_report
from scripts.preprocess_ddinter import pair_key


def test_brand_alias_is_normalised():
    item = normalise_medication("Coumadin")
    assert item.canonical_name == "warfarin" and item.normalized


def test_known_high_risk_ddi_is_flagged_with_evidence():
    result = analyze_medications("Warfarin\nAspirin")
    assert len(result.alerts) == 1 and result.alerts[0].severity == "high" and result.alerts[0].evidence


def test_food_rule_is_flagged():
    result = analyze_medications("Simvastatin", "grapefruit juice")
    assert len(result.alerts) == 1 and result.alerts[0].interaction_type == "drug-food"


def test_no_match_never_claims_safe():
    result = analyze_medications("Acetaminophen")
    assert not result.alerts and "not mean the combination is safe" in result.warnings[0]


def test_report_contains_reproducibility_and_disclaimer():
    result = analyze_medications("Warfarin\nAspirin")
    report = render_html_report(result, "Bounded explanation")
    assert result.run_id in report and "Do not start, stop, or change medication" in report


def test_pair_key_is_order_independent():
    assert pair_key("DDInter12", "DDInter3") == pair_key("DDInter3", "DDInter12")


def test_full_ddinter_lookup_finds_downloaded_pair():
    result = analyze_medications("Abacavir\nNaltrexone")
    assert len(result.alerts) == 1
    assert result.alerts[0].severity == "moderate"
    from adr_system.data import load_ddinter_lookup
    assert result.alerts[0].data_version == load_ddinter_lookup()[0]['version']


def test_unknown_medication_is_reported_without_false_alert():
    result = analyze_medications("Abacavir\nNotARealDrug")
    assert not result.alerts
    assert result.unsupported_medications == ["NotARealDrug"]
