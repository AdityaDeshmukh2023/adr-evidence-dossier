"""Dependency-light probability calibration and conservative selective prediction."""
from __future__ import annotations

import math


def softmax(logits, temperature: float = 1.0) -> list[list[float]]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive and finite.')
    output = []
    for row in logits:
        if len(row) != 3 or not all(math.isfinite(float(v)) for v in row):
            raise ValueError('Expected three finite source-severity logits.')
        maximum = max(float(v) for v in row)
        values = [math.exp((float(v) - maximum) / temperature) for v in row]
        total = sum(values)
        output.append([value / total for value in values])
    return output


def nll(probabilities, labels) -> float | None:
    if not labels:
        return None
    return -sum(math.log(max(float(p[y]), 1e-15)) for p, y in zip(probabilities, labels)) / len(labels)


def fit_temperature(logits, labels) -> float:
    if not labels or len(logits) != len(labels):
        raise ValueError('Calibration requires labeled logits.')
    # Scalar bounded golden-section optimization; no train/test data involved.
    left, right, ratio = -3.0, 3.0, (math.sqrt(5) - 1) / 2
    def loss(log_temperature):
        return nll(softmax(logits, math.exp(log_temperature)), labels)
    first, second = right - ratio * (right - left), left + ratio * (right - left)
    f_first, f_second = loss(first), loss(second)
    for _ in range(45):
        if f_first <= f_second:
            right, second, f_second = second, first, f_first
            first = right - ratio * (right - left)
            f_first = loss(first)
        else:
            left, first, f_first = first, second, f_second
            second = left + ratio * (right - left)
            f_second = loss(second)
    fitted = math.exp((left + right) / 2)
    return fitted if loss(math.log(fitted)) <= loss(0.0) else 1.0


def wilson_upper(errors: int, count: int, z: float = 1.959963984540054) -> float:
    if count <= 0 or errors < 0 or errors > count:
        raise ValueError('Invalid binomial error count.')
    proportion = errors / count
    denominator = 1 + z * z / count
    center = proportion + z * z / (2 * count)
    margin = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
    return (center + margin) / denominator


def choose_threshold(probabilities, labels, *, max_error: float = .1, minimum: int = 100) -> dict:
    if len(probabilities) != len(labels) or minimum < 1 or not 0 < max_error < 1:
        raise ValueError('Invalid threshold selection data.')
    ranked = sorted(((max(row), int(max(range(3), key=lambda i: row[i]) != label))
                     for row, label in zip(probabilities, labels)), reverse=True)
    best, errors, position = None, 0, 0
    while position < len(ranked):
        threshold = ranked[position][0]
        while position < len(ranked) and ranked[position][0] == threshold:
            errors += ranked[position][1]
            position += 1
        upper = wilson_upper(errors, position)
        if position >= minimum and upper <= max_error:
            best = {'threshold': threshold, 'accepted_count': position, 'errors': errors,
                    'wilson_95_upper_error': upper, 'coverage': position / len(ranked)}
    return {**(best or {'threshold': None, 'accepted_count': 0, 'coverage': 0,
                       'reason': 'no_calibration_threshold_qualifies'}),
            'minimum_accepted': minimum, 'maximum_upper_error': max_error,
            'sample_count': len(labels), 'adaptive_selection': True, 'formal_risk_guarantee': 'none',
            'interval_semantics': 'Selected-point Wilson statistic; adaptive threshold search is not uniform risk control.',
            'scope': 'Held-out source-label error, not clinical risk.'}


def probability_metrics(probabilities, labels, bins: int = 10) -> dict:
    if not labels:
        return {'count': 0, 'nll': None, 'brier': None, 'ece': None, 'reliability': []}
    brier = sum(sum((p[i] - int(label == i)) ** 2 for i in range(3))
                for p, label in zip(probabilities, labels)) / len(labels)
    reliability, ece = [], 0.0
    for index in range(bins):
        selected = [(max(row), int(max(range(3), key=lambda i: row[i]) == label))
                    for row, label in zip(probabilities, labels)
                    if min(int(max(row) * bins), bins - 1) == index]
        if not selected:
            continue
        confidence = sum(v[0] for v in selected) / len(selected)
        accuracy = sum(v[1] for v in selected) / len(selected)
        ece += len(selected) / len(labels) * abs(confidence - accuracy)
        reliability.append({'lower': index / bins, 'upper': (index + 1) / bins,
                            'count': len(selected), 'confidence': confidence, 'accuracy': accuracy})
    return {'count': len(labels), 'nll': nll(probabilities, labels), 'brier': brier,
            'ece': ece, 'reliability': reliability}

