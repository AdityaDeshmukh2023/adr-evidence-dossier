from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import quote

import requests

from .models import Evidence

TIMEOUT_SECONDS = 6


@lru_cache(maxsize=128)
def fetch_openfda_label(drug: str) -> tuple[list[Evidence], str]:
    """Fetch label evidence opportunistically; failure never blocks local analysis."""
    url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{quote(drug)}&limit=1"
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        result = response.json()["results"][0]
        excerpts = result.get("drug_interactions") or result.get("warnings") or []
        excerpt = " ".join(excerpts)[:700] if excerpts else "Current label record found; no interaction section was returned."
        return [Evidence("Current FDA drug label", url, excerpt, "openFDA drug labels",
                         datetime.now(timezone.utc).date().isoformat())], "live"
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return [], "unavailable"


@lru_cache(maxsize=128)
def fetch_pubchem_summary(drug: str) -> tuple[list[Evidence], str]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(drug)}/property/MolecularFormula,MolecularWeight/JSON"
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        props = response.json()["PropertyTable"]["Properties"][0]
        excerpt = f"Formula: {props.get('MolecularFormula', 'not available')}; molecular weight: {props.get('MolecularWeight', 'not available')}."
        return [Evidence("PubChem compound identity", url, excerpt, "PubChem",
                         datetime.now(timezone.utc).date().isoformat())], "live"
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return [], "unavailable"


def live_evidence_for(drugs: list[str]) -> tuple[dict[str, list[Evidence]], dict[str, str]]:
    evidence: dict[str, list[Evidence]] = {}
    status: dict[str, str] = {}
    for drug in sorted(set(drugs)):
        labels, label_status = fetch_openfda_label(drug)
        chemistry, chemistry_status = fetch_pubchem_summary(drug)
        evidence[drug] = labels + chemistry
        status[f"openFDA:{drug}"] = label_status
        status[f"PubChem:{drug}"] = chemistry_status
    return evidence, status
