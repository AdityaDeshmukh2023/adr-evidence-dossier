"""Execute actual PaddleOCR on grouped synthetic printed prescription images."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hybrid_benchmark import (evaluate_medications, split_for_group, synthetic_dataset,
                                     write_csv, write_outputs)
from adr_system.research import edit_distance, gold_ids


def _font(split: str, size: int):
    from PIL import ImageFont
    candidates = {
        'development': ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        'calibration': ('C:/Windows/Fonts/calibri.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'),
        'test': ('C:/Windows/Fonts/georgia.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'),
    }
    path = next((Path(p) for p in candidates[split] if Path(p).is_file()), None)
    if path is None:
        raise FileNotFoundError(f'No font for {split}; install a listed font to preserve font holdout.')
    return ImageFont.truetype(str(path), size), path


def _rotated_box(box, size, angle):
    cx, cy = size[0] / 2, size[1] / 2
    radians = math.radians(angle)
    corners = []
    for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
        dx, dy = x - cx, y - cy
        corners.append((cx + math.cos(radians) * dx + math.sin(radians) * dy,
                        cy - math.sin(radians) * dx + math.cos(radians) * dy))
    return [min(x for x, _ in corners), min(y for _, y in corners),
            max(x for x, _ in corners), max(y for _, y in corners)]


def synthetic_images(output: Path, groups: int = 40, seed: int = 31415) -> dict:
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
    output.mkdir(parents=True, exist_ok=True)
    source = synthetic_dataset(groups, seed)
    clean_cases = [case for case in source['cases'] if case['input_kind'] == 'clean']
    cases = []
    for index, base in enumerate(clean_cases):
        # OCR sizes stay within 2/5/10 so line detection is practical on CPU.
        entries = base['entries'][:(2, 5, 10)[index % 3]]
        from adr_system.research import source_reference
        gold_findings = source_reference(entries)
        split = split_for_group(index, strata=1)
        font, font_path = _font(split, 38)
        layout = 'flush_left' if split == 'development' else 'indented' if split == 'calibration' else 'right_column'
        x = {'flush_left': 65, 'indented': 165, 'right_column': 340}[layout]
        width, height = 1600, max(420, 100 + 90 * len(entries))
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        annotated = []
        rendered_lines = []
        for row, entry in enumerate(entries):
            text = f"{entry['text'].title()} 10 mg oral daily"
            y = 55 + 90 * row
            draw.text((x, y), text, font=font, fill='#161616')
            box = list(draw.textbbox((x, y), text, font=font))
            annotated.append({**entry, 'gold_bbox': box})
            rendered_lines.append(text)
        degraded = ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(1.1))).enhance(.42)
        variants = [('clean', image, 0), ('rotation_3_degrees', image.rotate(3, fillcolor='white'), 3),
                    ('blur_low_contrast', degraded, 0)]
        for variant, bitmap, angle in variants:
            case_id = f'HOCR-{index:04d}-{variant}'
            path = output / f'{case_id}.png'
            bitmap.save(path)
            cases.append({'id': case_id, 'group_id': f'OCR{index:04d}', 'split': split,
                          'input_kind': variant, 'size': len(entries), 'image': str(path.resolve()),
                          'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'font_name': font_path.name, 'font_sha256': hashlib.sha256(font_path.read_bytes()).hexdigest(),
                          'layout': layout, 'gold_text': '\n'.join(rendered_lines),
                          'entries': [{**entry, 'gold_bbox': _rotated_box(entry['gold_bbox'], image.size, angle)}
                                      for entry in annotated],
                          'gold_findings': gold_findings})
    return {'version': 'hybrid-ocr-synthetic-1.0', 'kind': 'synthetic_source_conformance',
            'modality': 'actual_image_ocr', 'base_groups': groups, 'seed': seed,
            'split_policy': '60/20/20 grouped splits; distinct font and layout per split.',
            'scope': 'Generated printed English medicine-section images; no handwriting or patient information.',
            'label_provenance': 'Rendered text/boxes and independent clean-identity source lookup.', 'cases': cases}


def validate_image_dataset(dataset: dict):
    if dataset.get('kind') != 'synthetic_source_conformance' or dataset.get('modality') != 'actual_image_ocr':
        raise ValueError('Only the declared generated-image benchmark is supported.')
    ids, groups, fonts, layouts = set(), {}, {}, {}
    for case in dataset['cases']:
        if case['id'] in ids:
            raise ValueError('Duplicate image case ID.')
        ids.add(case['id'])
        if groups.setdefault(case['group_id'], case['split']) != case['split']:
            raise ValueError('Related image variants leak across splits.')
        for key, buckets in (('font_sha256', fonts), ('layout', layouts)):
            if buckets.setdefault(case[key], case['split']) != case['split']:
                raise ValueError(f'{key} leaks across held-out image splits.')
        path = Path(case['image'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['image_sha256']:
            raise ValueError('Image bytes disagree with the dataset manifest.')


def _box_overlap(first, second):
    if first is None or second is None or len(first) != 4 or len(second) != 4:
        return 0.0
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    return intersection / max(1, area_a + area_b - intersection)


def align_for_scoring(medications, entries):
    """One-to-one geometry-only alignment; reference words never resolve OCR."""
    candidates = sorted((( _box_overlap(m.bbox, e['gold_bbox']), m.mention_id, index)
                         for m in medications for index, e in enumerate(entries)), reverse=True)
    matched_mentions, matched_entries, mapping, overlaps = set(), set(), {}, []
    for overlap, mention, index in candidates:
        if overlap < .15 or mention in matched_mentions or index in matched_entries:
            continue
        matched_mentions.add(mention)
        matched_entries.add(index)
        mapping[mention] = gold_ids(entries[index])
        overlaps.append(overlap)
    return mapping, {'boundary_matches': len(mapping), 'gold_entries': len(entries),
                     'detected_entries': len(medications), 'bbox_iou_sum': sum(overlaps)}


def evaluate_images(dataset: dict, output: Path, *, split: str = 'all') -> dict:
    from paddle_ocr import extract_document, medications_from_document
    from adr_system.provenance import manifest
    run_manifest = manifest()
    rows, diagnostics, documents = [], [], []
    selected = [c for c in dataset['cases'] if split == 'all' or c['split'] == split]
    for number, case in enumerate(selected, 1):
        start = perf_counter()
        error = None
        try:
            document = extract_document(case['image'])
            extraction_ms = (perf_counter() - start) * 1000
            normalization_start = perf_counter()
            medications = medications_from_document(document)
            normalization_ms = (perf_counter() - normalization_start) * 1000
            predicted_text = document['text']
            documents.append({'case_id': case['id'], 'document': document})
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            extraction_ms = (perf_counter() - start) * 1000
            normalization_ms = 0
            medications, predicted_text = [], ''
        mapping, boundaries = align_for_scoring(medications, case['entries'])
        case_rows = evaluate_medications(case, medications, gold_by_mention=mapping,
                                         gold_entry_count=len(case['entries']))
        for row in case_rows:
            row.update({'ocr_error': error, 'ocr_ms': extraction_ms,
                        'normalization_ms': normalization_ms,
                        'font_name': case['font_name'], 'layout': case['layout'],
                        'end_to_end_ms': extraction_ms + normalization_ms + row['latency_ms']})
        rows.extend(case_rows)
        reference = ' '.join(case['gold_text'].casefold().split())
        prediction = ' '.join(predicted_text.casefold().split())
        diagnostics.append({'case_id': case['id'], 'group_id': case['group_id'],
                            'split': case['split'], 'input_kind': case['input_kind'],
                            'font_name': case['font_name'], 'layout': case['layout'],
                            'error': error, 'ocr_ms': extraction_ms, **boundaries,
                            'character_edits': edit_distance(reference, prediction),
                            'reference_characters': len(reference),
                            'word_edits': edit_distance(reference.split(), prediction.split()),
                            'reference_words': len(reference.split())})
        if number % 10 == 0:
            print(json.dumps({'completed_images': number, 'total_images': len(selected)}), flush=True)
    if not rows:
        raise ValueError('No selected image cases.')
    summary = write_outputs(dataset, rows, output, run_manifest=run_manifest)
    write_csv(output / 'image_metrics.csv', diagnostics)
    (output / 'synthetic_ocr_outputs.jsonl').write_text(
        ''.join(json.dumps(item) + '\n' for item in documents), encoding='utf-8')
    total_chars = sum(d['reference_characters'] for d in diagnostics)
    total_words = sum(d['reference_words'] for d in diagnostics)
    summary.update({'actual_ocr_executed': True, 'images': len(diagnostics),
                    'ocr_failures': sum(d['error'] is not None for d in diagnostics),
                    'successful_ocr_images': sum(d['error'] is None for d in diagnostics),
                    'character_error_rate': sum(d['character_edits'] for d in diagnostics) / max(1, total_chars),
                    'word_error_rate': sum(d['word_edits'] for d in diagnostics) / max(1, total_words),
                    'entry_boundary_recall': sum(d['boundary_matches'] for d in diagnostics) /
                                             max(1, sum(d['gold_entries'] for d in diagnostics)),
                    'entry_boundary_precision': sum(d['boundary_matches'] for d in diagnostics) /
                                                max(1, sum(d['detected_entries'] for d in diagnostics)),
                    'ocr_scope': 'Actual PaddleOCR inference on generated printed images; held-out fonts/layouts.'})
    summary['execution_status'] = 'executed_with_errors' if summary['ocr_failures'] else 'executed'
    summary['limitations'].append('Geometry aligns mentions for scoring only; missed and extra OCR entries remain errors.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', type=int, default=40)
    parser.add_argument('--seed', type=int, default=31415)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/hybrid/ocr')
    parser.add_argument('--split', choices=('all', 'development', 'calibration', 'test'), default='all')
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.dataset.read_text(encoding='utf-8')) if args.dataset else synthetic_images(
        args.output / 'images', args.groups, args.seed)
    validate_image_dataset(dataset)
    (args.output / 'dataset.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')
    if args.generate_only:
        print(json.dumps({'execution_status': 'images_generated_only', 'images': len(dataset['cases']),
                          'groups': dataset['base_groups'], 'actual_ocr_executed': False}))
        return 0
    summary = evaluate_images(dataset, args.output, split=args.split)
    print(json.dumps({k: summary[k] for k in ('images', 'ocr_failures', 'character_error_rate',
                                             'word_error_rate', 'entry_boundary_recall')}, indent=2))
    return int(summary['ocr_failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
