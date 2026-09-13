"""Event-level P/T wave delineation evaluation (paper Section 4.4, Table 4).

Because sample-wise Macro F1 does not directly assess individual event
detection, this module performs one-to-one matching between predicted and
reference wave *events* (contiguous runs of a non-background class). A match
requires the predicted and reference event onsets to fall within 25 samples
(100 ms at 250 Hz) of one another; each prediction and each reference event
may participate in at most one match. Unmatched predictions/references count
as false positives/false negatives respectively.
"""
import numpy as np

MATCH_TOLERANCE_SAMPLES = 25  # 100 ms at 250 Hz


def extract_events(labels, class_index):
    """Return a list of (onset, offset) sample-index pairs for one wave class."""
    is_class = labels == class_index
    events = []
    start = None
    for i, flag in enumerate(is_class):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(labels) - 1))
    return events


def match_events(pred_events, ref_events, tolerance=MATCH_TOLERANCE_SAMPLES):
    """Greedy nearest-onset one-to-one matching within `tolerance` samples.

    Returns (true_positives, false_positives, false_negatives) as event-index lists.
    """
    unmatched_ref = list(range(len(ref_events)))
    tp_pairs = []
    fp = []
    for p_idx, (p_on, _) in enumerate(pred_events):
        best_j, best_dist = None, None
        for j in unmatched_ref:
            dist = abs(ref_events[j][0] - p_on)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_j, best_dist = j, dist
        if best_j is not None:
            tp_pairs.append((p_idx, best_j))
            unmatched_ref.remove(best_j)
        else:
            fp.append(p_idx)
    fn = unmatched_ref
    return tp_pairs, fp, fn


def event_level_metrics(y_true, y_pred, class_index, tolerance=MATCH_TOLERANCE_SAMPLES):
    """Sensitivity / PPV / F1 for one wave class over one record (or pooled array)."""
    ref_events = extract_events(y_true, class_index)
    pred_events = extract_events(y_pred, class_index)
    tp_pairs, fp, fn = match_events(pred_events, ref_events, tolerance)
    tp = len(tp_pairs)
    sensitivity = tp / len(ref_events) if ref_events else float('nan')
    ppv = tp / len(pred_events) if pred_events else float('nan')
    f1 = 2 * tp / (2 * tp + len(fp) + len(fn)) if (tp + len(fp) + len(fn)) else float('nan')
    return {'tp': tp, 'fp': len(fp), 'fn': len(fn), 'sensitivity': sensitivity, 'ppv': ppv, 'f1': f1}
