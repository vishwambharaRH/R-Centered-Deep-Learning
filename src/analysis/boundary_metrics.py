"""Boundary-level (onset/offset) delineation error in milliseconds (paper Section 4.5, Table 5).

Computed only for successfully matched events (see `event_metrics.match_events`):
predicted onset/offset boundaries are compared with the corresponding reference
boundaries. Positive signed error indicates a later prediction; negative
indicates earlier. Reported as mean absolute error (MAE) per wave/boundary.
"""
import numpy as np

from .event_metrics import extract_events, match_events


def boundary_errors_ms(y_true, y_pred, class_index, sampling_rate_hz, tolerance=25):
    """Signed onset/offset errors (ms) for every matched event of one class."""
    ref_events = extract_events(y_true, class_index)
    pred_events = extract_events(y_pred, class_index)
    tp_pairs, _, _ = match_events(pred_events, ref_events, tolerance)

    ms_per_sample = 1000.0 / sampling_rate_hz
    onset_errors, offset_errors = [], []
    for p_idx, r_idx in tp_pairs:
        p_on, p_off = pred_events[p_idx]
        r_on, r_off = ref_events[r_idx]
        onset_errors.append((p_on - r_on) * ms_per_sample)
        offset_errors.append((p_off - r_off) * ms_per_sample)
    return np.asarray(onset_errors), np.asarray(offset_errors)


def boundary_mae(y_true, y_pred, class_index, sampling_rate_hz, tolerance=25):
    onset_errors, offset_errors = boundary_errors_ms(y_true, y_pred, class_index, sampling_rate_hz, tolerance)
    onset_mae = float(np.mean(np.abs(onset_errors))) if len(onset_errors) else float('nan')
    offset_mae = float(np.mean(np.abs(offset_errors))) if len(offset_errors) else float('nan')
    return {'onset_mae_ms': onset_mae, 'offset_mae_ms': offset_mae}
