"""Import-safe compatibility wrapper for optional openFDA label evidence."""
from adr_system.evidence import fetch_openfda_label


def check_drug_label(drug: str):
    return fetch_openfda_label(drug)
