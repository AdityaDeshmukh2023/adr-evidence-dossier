"""Resumable PubChem identity mapping. No interaction labels affect chemistry."""
from __future__ import annotations

import hashlib
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .io import read_json, write_json, object_hash

PUG = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
FINGERPRINT = {'method': 'Morgan', 'radius': 2, 'bits': 2048, 'include_chirality': True}


def chemistry(smiles: str) -> dict:
    """Reject disconnected structures rather than silently stripping salts/mixtures."""
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError('Install requirements-ml.txt in an optional ML environment.') from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid_smiles')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('disconnected_structure')
    if mol.GetNumHeavyAtoms() > 150:
        raise ValueError('unsupported_large_structure')
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                          includeChirality=True)
    return {'canonical_smiles': canonical, 'standardized_inchikey': Chem.MolToInchiKey(mol),
            'fingerprint_bits': list(generator.GetFingerprint(mol).GetOnBits()),
            'rdkit_version': rdBase.rdkitVersion, 'fingerprint': FINGERPRINT}


class PubChemMapper:
    def __init__(self, *, timeout: float = 20, interval: float = .35, limiter=None):
        import requests
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Healthier-DDI-research/1.0 (academic ingredient mapping)'
        self.timeout, self.interval, self.last_request = timeout, interval, 0.0
        self.limiter = limiter

    def _get(self, url: str) -> dict:
        if self.limiter is not None:
            self.limiter()
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request = time.monotonic()
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def map(self, ingredient_id: str, name: str) -> dict:
        url = f'{PUG}/compound/name/{quote(name, safe="")}/property/IsomericSMILES,InChIKey/JSON?name_type=complete'
        row = {'ingredient_id': ingredient_id, 'name': name, 'query_url': url,
               'retrieved_at': datetime.now(timezone.utc).isoformat(), 'status': 'unmapped'}
        try:
            payload = self._get(url)
            properties_list = payload.get('PropertyTable', {}).get('Properties', [])
            ids = sorted(set(p.get('CID') for p in properties_list if p.get('CID') is not None))
            row['candidate_cids'] = ids
            row['identity_response_sha256'] = object_hash(payload)
            if len(ids) != 1:
                row['reason'] = 'ambiguous_pubchem_identity' if ids else 'pubchem_identity_absent'
                return row
            row['cid'] = ids[0]
            properties_payload = payload
            properties = properties_list[0]
            smiles = properties.get('SMILES') or properties.get('IsomericSMILES')
            if not smiles:
                row['reason'] = 'missing_pubchem_smiles'
                return row
            row.update({'property_url': url, 'pubchem_inchikey': properties.get('InChIKey'),
                        'property_response_sha256': object_hash(properties_payload),
                        'raw_smiles': smiles})
            try:
                row.update(chemistry(smiles))
            except ValueError as exc:
                row['reason'] = str(exc)
                return row
            if row['standardized_inchikey'] != row['pubchem_inchikey']:
                row['reason'] = 'rdkit_pubchem_inchikey_mismatch'
                return row
            row.update(status='mapped', identity_policy='one_complete_name_CID; one connected molecule')
        except Exception as exc:
            # Persist failed network/parse attempts separately from identity exclusions.
            if isinstance(exc, RuntimeError):
                raise
            status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status_code == 404:
                row.update(status='unmapped', reason='pubchem_identity_absent', http_status=404)
            else:
                row.update(status='error', reason=type(exc).__name__, error=str(exc)[:400],
                           http_status=status_code)
        return row


def prepare_molecules(catalog: dict[str, str], output: str | Path, *, limit: int | None = None,
                      retry_failed: bool = False, timeout: float = 20,
                      max_consecutive_errors: int = 5, workers: int = 4, notify=None) -> dict:
    output = Path(output)
    saved = read_json(output) if output.exists() else {'schema': 'molecule-cache-2', 'records': {}}
    records = saved['records']
    selected = sorted(catalog.items())[:limit] if limit is not None else sorted(catalog.items())
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four workers; the shared limit is below three requests/second.')
    pending_rows = [(i, n) for i, n in selected if not (records.get(i) and records[i].get('name') == n
                     and (records[i].get('status') != 'error' or not retry_failed)
                     and (records[i].get('status') != 'mapped' or
                          records[i].get('standardized_inchikey') == records[i].get('pubchem_inchikey')))]
    lock, last_request, local = threading.Lock(), [0.0], threading.local()
    def limiter():
        with lock:
            delay = .35 - (time.monotonic() - last_request[0])
            if delay > 0:
                time.sleep(delay)
            last_request[0] = time.monotonic()
    def fetch(item):
        if not hasattr(local, 'mapper'):
            local.mapper = PubChemMapper(timeout=timeout, interval=0, limiter=limiter)
        return local.mapper.map(*item)
    failures, attempts = 0, 0
    iterator = iter(pending_rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, row): row for row in [next(iterator, None) for _ in range(workers)] if row}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                ident, name = futures.pop(future)
                row = future.result()
                records[ident] = row
                failures = failures + 1 if row['status'] == 'error' else 0
                attempts += 1
                saved.update(schema='molecule-cache-2', catalog_size=len(catalog), requested_size=len(selected), fingerprint=FINGERPRINT,
                             records=records, catalog_sha256=object_hash(catalog),
                             status='in_progress', updated_at=datetime.now(timezone.utc).isoformat())
                write_json(output, saved)
                if notify:
                    notify({'id': ident, 'status': row['status'], 'attempts': attempts})
                if failures < max_consecutive_errors:
                    next_row = next(iterator, None)
                    if next_row:
                        futures[executor.submit(fetch, next_row)] = next_row
            if failures >= max_consecutive_errors:
                for future in futures:
                    future.cancel()
                break
    mapped = sum(records.get(i, {}).get('status') == 'mapped' for i, _ in selected)
    errors = sum(records.get(i, {}).get('status') == 'error' for i, _ in selected)
    saved.update(status='network_interrupted' if failures >= max_consecutive_errors else 'complete',
                 mapped_requested=mapped, errors_requested=errors,
                 unattempted_requested=sum(i not in records for i, _ in selected), attempts_this_run=attempts)
    write_json(output, saved)
    return saved

