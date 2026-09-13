"""Sample-wise evaluation: Macro F1 across {Background, P, T}.

Predictions from all generated windows in a partition are pooled and compared
against the corresponding window labels (Section 3.2.3): "continuous full-record
predictions were not reconstructed."
"""
from sklearn.metrics import accuracy_score, classification_report, f1_score

CLASS_NAMES = ['Background', 'P', 'T']


def macro_f1(y_true, y_pred):
    return f1_score(y_true.ravel(), y_pred.ravel(), labels=[0, 1, 2], average='macro', zero_division=0)


def full_report(y_true, y_pred):
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': macro_f1(y_true, y_pred),
        'weighted_f1': f1_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0),
        'per_class': classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        ),
    }
