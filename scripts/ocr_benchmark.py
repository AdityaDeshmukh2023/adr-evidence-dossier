"""Actual image OCR evaluation; never substitute text corruption for image results."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synthetic_images(output):
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    from adr_system.research import source_reference
    fonts = [Path('C:/Windows/Fonts/arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font = next((ImageFont.truetype(str(p), 48) for p in fonts if p.exists()), ImageFont.load_default(size=48))
    cases = []
    for number, names in enumerate([['warfarin', 'aspirin'], ['abacavir', 'naltrexone']]):
        image = Image.new('RGB', (1400, 500), 'white')
        draw = ImageDraw.Draw(image)
        for i, name in enumerate(names):
            draw.text((70, 100 + i * 160), name.title(), font=font, fill='black')
        variants = {'clean': image, 'rotation_3_degrees': image.rotate(3, fillcolor='white'),
                    'blur': image.filter(ImageFilter.GaussianBlur(1.4)),
                    'low_contrast': ImageEnhance.Contrast(image).enhance(.35)}
        for variant, bitmap in variants.items():
            path = output / f'printed_{number}_{variant}.png'
            bitmap.save(path)
            cases.append({'id': path.stem, 'group_id': f'image-{number}', 'kind': 'printed', 'variant': variant,
                          'image': str(path.resolve()), 'gold_text': '\n'.join(n.title() for n in names),
                          'gold_names': names, 'gold_findings': source_reference([{'gold_names': [n]} for n in names])})
    return {'kind': 'synthetic', 'image_cases': cases}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from paddle_ocr import extract_document, medications_from_document
    from adr_system.engine import analyze_structured
    from adr_system.research import edit_distance, finding_set
    from adr_system.terminology import exact
    from adr_system.provenance import manifest
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ocr_benchmark')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.dataset.read_text(encoding='utf-8')) if args.dataset else synthetic_images(args.output)
    if dataset['kind'] != 'synthetic' and not dataset.get('permission_confirmed'):
        raise ValueError('Permission must be documented for real prescription images.')
    rows, documents = [], []
    for case in dataset['image_cases']:
        if dataset['kind'] != 'synthetic' and case.get('annotation_status') != 'reviewed':
            raise ValueError('Real image cases need reviewed reference annotations.')
        start = perf_counter()
        error = None
        try:
            image_path = Path(case['image'])
            if not image_path.is_absolute() and args.dataset:
                image_path = args.dataset.parent / image_path
            document = extract_document(str(image_path))
            result = analyze_structured(medications_from_document(document))
            predicted_text = document['text']
            ids = {i for m in result.medications for i in m.ingredient_ids}
            findings = finding_set(result)
            documents.append({'case_id': case['id'], 'document': document, 'result': result.to_dict()})
        except Exception as exc:
            error = type(exc).__name__
            predicted_text, ids, findings = '', set(), set()
        gold_ids = {i for n in case['gold_names'] for i in exact(n)[0].ingredient_ids}
        gold_findings = {(tuple(sorted(f['ingredient_ids'])), f['severity']) for f in case['gold_findings']}
        reference = ' '.join(case['gold_text'].casefold().split())
        prediction = ' '.join(predicted_text.casefold().split())
        rows.append({'case_id': case['id'], 'group_id': case['group_id'], 'kind': case['kind'],
                     'variant': case.get('variant', 'real_image'), 'error': error,
                     'character_edits': edit_distance(reference, prediction), 'reference_characters': len(reference),
                     'word_edits': edit_distance(reference.split(), prediction.split()), 'reference_words': len(reference.split()),
                     'ingredient_correct': len(ids & gold_ids), 'ingredient_total': len(gold_ids),
                     'ingredient_extra': len(ids - gold_ids), 'finding_correct': len(findings & gold_findings),
                     'finding_total': len(gold_findings), 'finding_extra': len(findings - gold_findings),
                     'latency_seconds': perf_counter() - start})
    frame = pd.DataFrame(rows)
    frame['cer'] = frame.character_edits / frame.reference_characters.clip(lower=1)
    frame['wer'] = frame.word_edits / frame.reference_words.clip(lower=1)
    frame.to_csv(args.output / 'image_metrics.csv', index=False)
    # Never persist raw real-image OCR text by default.
    if dataset['kind'] == 'synthetic':
        (args.output / 'synthetic_outputs.json').write_text(json.dumps(documents, indent=2), encoding='utf-8')
    summary = {'dataset_kind': dataset['kind'], 'images': len(rows), 'groups': int(frame.group_id.nunique()),
               'failures': int(frame.error.notna().sum()),
               'character_error_rate': float(frame.character_edits.sum() / max(1, frame.reference_characters.sum())),
               'word_error_rate': float(frame.word_edits.sum() / max(1, frame.reference_words.sum())),
               'ingredient_recall': float(frame.ingredient_correct.sum() / max(1, frame.ingredient_total.sum())),
               'finding_recall': float(frame.finding_correct.sum() / max(1, frame.finding_total.sum())),
               'manifest': manifest(), 'scope': 'Actual OCR execution; synthetic images do not establish real prescription performance.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 4))
    frame.groupby('variant')[['cer', 'wer']].mean().plot.bar(ax=ax, color=['#24756c', '#bc8734'])
    ax.set(title=f'{dataset["kind"]} image OCR errors', ylabel='Mean error rate')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout(); fig.savefig(args.output / 'ocr_error_rates.png', dpi=220); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'manifest'}, indent=2))
    return int(summary['failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
