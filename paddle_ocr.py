"""Structured PaddleOCR 3.x adapter; inference is local and initialization cached."""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('FLAGS_use_mkldnn', '0')
_LOCK = threading.Lock()
MODEL_NAMES = ('PP-OCRv5_mobile_det', 'en_PP-OCRv5_mobile_rec')


@lru_cache(maxsize=2)
def _create_ocr(lang: str = 'en'):
    if lang != 'en':
        raise ValueError('This evaluated configuration supports English only.')
    from paddleocr import PaddleOCR
    import paddle
    paddle.set_flags({'FLAGS_use_mkldnn': False})
    cache = Path(os.environ.get('PADDLE_PDX_CACHE_HOME', str(Path.home() / '.paddlex'))) / 'official_models'
    local = {}
    for option, model in zip(('text_detection_model_dir', 'text_recognition_model_dir'), MODEL_NAMES):
        directory = cache / model
        if (directory / 'inference.pdiparams').exists():
            local[option] = str(directory)
    return PaddleOCR(text_detection_model_name=MODEL_NAMES[0], text_recognition_model_name=MODEL_NAMES[1],
                     use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=False, device='cpu', enable_mkldnn=False, **local)


def parse_ocr_results(results) -> list[dict]:
    lines = []
    for page, result in enumerate(results or []):
        texts, scores = result.get('rec_texts', []), result.get('rec_scores', [])
        boxes = result.get('rec_boxes')
        polys = result.get('rec_polys')
        if len(texts) != len(scores):
            raise ValueError('OCR text and score counts differ.')
        for i, text in enumerate(texts):
            if boxes is not None and len(boxes) > i:
                box = tuple(float(x) for x in boxes[i])
            elif polys is not None and len(polys) > i:
                points = polys[i]
                box = (min(float(p[0]) for p in points), min(float(p[1]) for p in points),
                       max(float(p[0]) for p in points), max(float(p[1]) for p in points))
            else:
                box = None
            if str(text).strip():
                lines.append({'text': str(text), 'score': float(scores[i]), 'bbox': box, 'page': page})
    # Preserve the engine's reading order; geometry is retained for review.
    return lines


def extract_document(img_path: str, lang: str = 'en') -> dict:
    from PIL import Image, ImageOps
    with Image.open(img_path) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError('Image exceeds the 20 megapixel limit.')
        normalized = ImageOps.exif_transpose(image).convert('RGB')
        import numpy as np
        pixels = np.asarray(normalized)
    with _LOCK:
        results = _create_ocr(lang).predict(pixels)
    lines = parse_ocr_results(results)
    return {'lines': lines, 'text': '\n'.join(line['text'] for line in lines),
            'image_sha256': hashlib.sha256(Path(img_path).read_bytes()).hexdigest(),
            'models': list(MODEL_NAMES), 'language': lang, 'preprocessing': 'EXIF transpose; RGB',
            'checkpoint_hashes': checkpoint_hashes()}


@lru_cache(maxsize=1)
def checkpoint_hashes() -> dict:
    cache = Path(os.environ.get('PADDLE_PDX_CACHE_HOME', str(Path.home() / '.paddlex')))
    hashes = {}
    for model in MODEL_NAMES:
        directory = cache / 'official_models' / model
        if directory.exists():
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.yml', '.yaml', '.pdiparams', '.pdparams'):
                    hashes[f'{model}/{path.relative_to(directory).as_posix()}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def medications_from_document(document: dict):
    from adr_system.normalization import parse_medication_lines, MAX_MEDICATIONS
    medications = []
    for line in document['lines']:
        for med in parse_medication_lines(line['text']):
            # Low recognition scores always require verification, even when the
            # OCR string happens to be an exact dictionary word.
            low = line['score'] < .90
            medications.append(replace(med, mention_id=f'm{len(medications) + 1}',
                bbox=tuple(line['bbox']) if line['bbox'] else None, ocr_score=line['score'],
                normalized=med.normalized and not low,
                ingredient_ids=() if low else med.ingredient_ids,
                ingredient_names=() if low else med.ingredient_names,
                canonical_name=None if low else med.canonical_name,
                status='ambiguous' if low and med.candidates else med.status))
    if len(medications) > MAX_MEDICATIONS:
        raise ValueError('Too many OCR entries; crop the medication section and retry.')
    return medications


def extract_text_from_image(img_path: str, lang: str = 'en') -> str:
    return extract_document(img_path, lang)['text']
