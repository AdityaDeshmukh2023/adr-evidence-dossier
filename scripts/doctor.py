"""Read-only installation checks. Never prints environment secrets."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    packages = {}
    for name in ('streamlit', 'pytest', 'paddleocr', 'paddlepaddle', 'groq', 'pandas', 'matplotlib',
                 'torch', 'rdkit', 'scikit-learn', 'shap', 'torch-geometric'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    from adr_system.engine import analyze_medications
    result = analyze_medications('Warfarin 5 mg Aspirin 75 mg')
    cache = Path(os.environ.get('PADDLE_PDX_CACHE_HOME', str(Path.home() / '.paddlex'))) / 'official_models'
    models = {}
    for name in ('PP-OCRv5_mobile_det', 'en_PP-OCRv5_mobile_rec'):
        try:
            models[name] = (cache / name / 'inference.pdiparams').exists()
        except PermissionError:
            models[name] = 'cache inaccessible in this execution context'
    from dotenv import load_dotenv
    load_dotenv()
    output = {'python': sys.version.split()[0], 'packages': packages, 'ocr_models_cached': models,
              'groq_key_configured': bool(os.getenv('GROQ_API_KEY')),
              'source_snapshot_present': (ROOT / 'data/processed_ddinter/ddinter_lookup.json').exists(),
              'biology': result.hybrid_status.get('biology'),
              'research_model_directory': os.getenv('HYBRID_MODEL_DIR', 'artifacts/ml/deployment'),
              'core_smoke_passed': len(result.alerts) == 1 and result.alerts[0].severity == 'high'}
    try:
        from adr_system.ml.inference import bundle_manifest
        output['research_model'] = bundle_manifest(os.getenv('HYBRID_MODEL_DIR', str(ROOT / 'artifacts/ml/deployment')))
    except ImportError:
        output['research_model'] = {'status': 'unavailable', 'reason': 'Optional ML inference code/profile unavailable.'}
    except (OSError, ValueError, RuntimeError) as error:
        output['research_model'] = {'status': 'incompatible', 'reason': str(error)[:200]}
    print(json.dumps(output, indent=2))
    return int(not output['core_smoke_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
