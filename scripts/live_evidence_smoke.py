"""Explicit network check using synthetic public ingredient names only."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from adr_system.engine import analyze_medications
    from adr_system.evidence import enrich_live
    result = enrich_live(analyze_medications('Warfarin\nAspirin'), refresh=True)
    summary = {'scope': 'Connectivity and record attachment only; not clinical evidence validation.',
               'source_status': result.source_status,
               'evidence': [{'purpose': e.purpose, 'document_id': e.document_id, 'version': e.document_version,
                             'excerpt_sha256': e.excerpt_hash} for e in result.alerts[0].evidence]}
    output = ROOT / 'artifacts/live_evidence'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
