from __future__ import annotations

import re
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import Evidence
from .terminology import catalog, exact

TIMEOUT_SECONDS = 6
_CACHE: dict = {}
_LOCK = threading.Lock()


def _get_json(url: str, params=None):
    with requests.Session() as session:
        retry = Retry(total=1, backoff_factor=.2, status_forcelist=(429, 502, 503, 504), respect_retry_after_header=False)
        session.mount('https://', HTTPAdapter(max_retries=retry))
        response = session.get(url, params=params, timeout=(3, TIMEOUT_SECONDS))
        response.raise_for_status()
        return response.json()


def clear_evidence_cache():
    with _LOCK:
        _CACHE.clear()


def _cached(key, fetch):
    with _LOCK:
        entry = _CACHE.get(key)
        if entry and entry[0] > time.monotonic():
            return entry[1]
    value = fetch()
    with _LOCK:
        if len(_CACHE) >= 256:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = (time.monotonic() + (3600 if value[1] == 'live' else 30), value)
    return value


def fetch_openfda_label(drug: str, partners: tuple[str, ...] = ()) -> tuple[list[Evidence], str]:
    def fetch():
        try:
            def synonyms(name):
                matches = exact(name)
                if len(matches) != 1:
                    return [name]
                ids = matches[0].ingredient_ids
                return [term for term, options in catalog()[0].items() if any(c.ingredient_ids == ids for c in options)]
            search = ' OR '.join(f'openfda.generic_name:"{re.sub(r"[^\w\s-]", " ", term)}"' for term in synonyms(drug))
            partner_terms = [term for partner in partners for term in synonyms(partner)]
            data = _get_json('https://api.fda.gov/drug/label.json', {
                'search': search, 'limit': 3, 'sort': 'effective_time:desc'})
            evidence = []
            for label in data.get('results', []):
                set_id = label.get('set_id')
                if not set_id:
                    continue
                sections = label.get('drug_interactions', [])
                passages = [p.strip() for section in sections for p in re.split(r'(?<=[.!?])\s+|\n+', section) if p.strip()]
                for passage in passages:
                    if not any(re.search(r'(?<!\w)' + re.escape(partner) + r'(?!\w)', passage, re.I) for partner in partner_terms):
                        continue
                    # Preserve complete relevant sentences; never take an arbitrary
                    # prefix of a label and call it pair-specific evidence.
                    if len(passage) > 3000:
                        continue
                    evidence.append(Evidence('Product label interaction passage',
                        f'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={quote(set_id)}',
                        passage, 'openFDA label', datetime.now(timezone.utc).isoformat(),
                        document_id=set_id, document_version=str(label.get('version', label.get('effective_time', ''))),
                        section='drug_interactions', purpose='supporting_label_passage'))
                    if len(evidence) >= 5:
                        break
            return evidence[:5], 'live' if evidence else 'no relevant passage'
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            logging.getLogger(__name__).warning('evidence_unavailable provider=openFDA error_type=%s', type(error).__name__)
            return [], 'unavailable'
    return _cached(('label', drug, partners), fetch)


def fetch_pubchem_summary(drug: str) -> tuple[list[Evidence], str]:
    def fetch():
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(drug, safe="")}/property/MolecularFormula,MolecularWeight/JSON'
        try:
            props = _get_json(url)['PropertyTable']['Properties'][0]
            cid = str(props['CID'])
            text = f"Formula: {props.get('MolecularFormula', 'unavailable')}; molecular weight: {props.get('MolecularWeight', 'unavailable')}. Identity information only."
            return [Evidence('PubChem compound identity', f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}',
                text, 'PubChem', datetime.now(timezone.utc).isoformat(), document_id=cid, purpose='identity_only')], 'live'
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as error:
            logging.getLogger(__name__).warning('evidence_unavailable provider=PubChem error_type=%s', type(error).__name__)
            return [], 'unavailable'
    return _cached(('pubchem', drug), fetch)


def live_evidence_for(drugs: list[str]):
    drugs = sorted(set(drugs))[:30]
    names = catalog()[1]
    def fetch(drug):
        matches = exact(drug)
        ident = matches[0].ingredient_ids[0] if len(matches) == 1 else drug
        partners = tuple(d for d in drugs if d != drug)
        labels, ls = fetch_openfda_label(drug, partners)
        chemistry, cs = fetch_pubchem_summary(drug)
        return ident, labels + chemistry, {f'openFDA:{names.get(ident, drug)}': ls, f'PubChem:{names.get(ident, drug)}': cs}
    evidence, statuses = {}, {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for ident, items, status in executor.map(fetch, drugs):
            evidence[ident] = items
            statuses.update(status)
    return evidence, statuses


def enrich_live(result, *, refresh: bool = False):
    if refresh:
        clear_evidence_cache()
    names = catalog()[1]
    ids = sorted({i for m in result.medications for i in m.ingredient_ids})
    evidence, statuses = live_evidence_for([names[i] for i in ids])
    result.source_status.update(statuses)
    for alert in result.alerts:
        existing = {e.evidence_id: e for e in alert.evidence if e.source not in ('openFDA label', 'PubChem')}
        for ident in alert.ingredient_ids:
            for item in evidence.get(ident, []):
                partner_terms = [term for term, choices in catalog()[0].items()
                                 if any(other != ident and other in choice.ingredient_ids
                                        for choice in choices for other in alert.ingredient_ids)]
                if item.purpose == 'identity_only' or any(
                    re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', item.excerpt, re.I)
                    for term in partner_terms):
                    existing[item.evidence_id] = item
        alert.evidence = list(existing.values())
    return result
