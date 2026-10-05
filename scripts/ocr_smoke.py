"""Generate a synthetic printed prescription and run actual local OCR."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from PIL import Image, ImageDraw, ImageFont
    from paddle_ocr import extract_document, medications_from_document
    from adr_system.engine import analyze_structured
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ocr_smoke')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (1400, 500), 'white')
    fonts = [Path('C:/Windows/Fonts/arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font = next((ImageFont.truetype(str(p), 54) for p in fonts if p.exists()), ImageFont.load_default(size=54))
    draw = ImageDraw.Draw(image)
    for y, line in [(100, 'Warfarin 5 mg'), (250, 'Aspirin 75 mg')]:
        draw.text((70, y), line, fill='black', font=font)
    path = args.output / 'synthetic_prescription.png'
    image.save(path)
    start = perf_counter()
    try:
        document = extract_document(str(path))
        result = analyze_structured(medications_from_document(document))
        passed = len(result.medications) == 2 and any(a.severity == 'high' for a in result.alerts)
        output = {'passed': passed, 'document': document, 'result': result.to_dict(),
                  'elapsed_seconds': perf_counter() - start, 'scope': 'One synthetic printed-image integration check; not OCR accuracy.'}
    except Exception as error:
        output = {'passed': False, 'error_type': type(error).__name__, 'error': str(error),
                  'scope': 'OCR integration unavailable; no recognition performance claimed.'}
    (args.output / 'result.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k not in ('document', 'result')}, indent=2))
    return int(not output['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
