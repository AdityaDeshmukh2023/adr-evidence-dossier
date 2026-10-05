"""Report source-label prediction metrics separately from clinical outcomes."""
from __future__ import annotations

from .calibration import probability_metrics, choose_threshold


def classification_metrics(probabilities, labels, *, threshold=None):
    if not labels:
        return {'count': 0, 'macro_f1': None, 'per_class': {}, 'high_pr_auc': None}
    import numpy as np
    from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
    predictions = [max(range(3), key=lambda i: row[i]) for row in probabilities]
    report = classification_report(labels, predictions, labels=[0, 1, 2],
                                   target_names=['high', 'moderate', 'low'], output_dict=True, zero_division=0)
    truth_high = [int(label == 0) for label in labels]
    pr_auc = float(average_precision_score(truth_high, [row[0] for row in probabilities])) if any(truth_high) else None
    output = {**probability_metrics(probabilities, labels), 'macro_f1': report['macro avg']['f1-score'],
              'per_class': {name: report[name] for name in ('high', 'moderate', 'low')},
              'high_pr_auc': pr_auc, 'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist()}
    accepted = [i for i, row in enumerate(probabilities) if threshold is not None and max(row) >= threshold]
    output['selective'] = {'threshold': threshold, 'accepted_count': len(accepted),
                           'coverage': len(accepted) / len(labels),
                           'error_rate': (sum(predictions[i] != labels[i] for i in accepted) / len(accepted)
                                          if accepted else None)}
    output['selective']['per_class'] = {}
    for label, name in enumerate(('high', 'moderate', 'low')):
        relevant = [i for i, value in enumerate(labels) if value == label]
        class_accepted = [i for i in accepted if labels[i] == label]
        output['selective']['per_class'][name] = {'source_count': len(relevant),
                                                 'accepted_count': len(class_accepted),
                                                 'coverage': len(class_accepted) / len(relevant) if relevant else None,
                                                 'error_rate': sum(predictions[i] != label for i in class_accepted) / len(class_accepted) if class_accepted else None}
    output['coverage_error_curve'] = []
    for value in (.4, .5, .6, .7, .8, .9, .95):
        chosen = [i for i, row in enumerate(probabilities) if max(row) >= value]
        output['coverage_error_curve'].append({'threshold': value, 'coverage': len(chosen) / len(labels),
                                               'error_rate': sum(predictions[i] != labels[i] for i in chosen) / len(chosen) if chosen else None})
    return output
