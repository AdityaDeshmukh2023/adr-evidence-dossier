from __future__ import annotations

import hashlib
import platform
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from .data import ROOT, load_json, load_ddinter_lookup


@lru_cache(maxsize=1)
def manifest() -> dict:
    artifacts = {name: load_json(name)[1] for name in
                 ('terminology.json', 'ddinter_demo_snapshot.json', 'drug_food_rules.json')}
    artifacts['ddinter_lookup.json'] = load_ddinter_lookup()[1]
    biological = ROOT / 'data/biological_roles.json'
    artifacts['biological_roles.json'] = hashlib.sha256(biological.read_bytes()).hexdigest() if biological.exists() else None
    code = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'adr_system').rglob('*.py'))}
    code.update({str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((ROOT / 'legacy/frozen_v1/adr_system').glob('*.py'))})
    for path in ('paddle_ocr.py', 'llm.py', *[str(p.relative_to(ROOT)).replace('\\', '/') for p in sorted((ROOT / 'scripts').glob('*.py'))],
                 *[p.name for p in sorted(ROOT.glob('requirements*.txt'))],
                 *[p.name for p in sorted(ROOT.glob('requirements*.lock'))]):
        code[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    packages = {}
    for name in ('streamlit', 'paddleocr', 'paddlepaddle', 'groq', 'pandas', 'torch', 'rdkit', 'shap', 'torch-geometric'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'not installed'
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                  text=True, timeout=3, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = 'unavailable'
    return {'artifacts': artifacts, 'code_sha256': code, 'git_revision': revision,
            'python': platform.python_version(), 'packages': packages,
            'algorithm': 'hybrid-screening-v1', 'evidence_graph_schema': '1.1',
            'candidate_limit': 5, 'candidate_min_similarity': .60,
            'scope': 'Known source records; not patient-specific risk prediction'}
