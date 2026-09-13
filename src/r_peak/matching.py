"""Reference R-peak extraction and one-to-one matching against detected peaks."""
import numpy as np

MATCH_TOLERANCE_SAMPLES_100MS_AT_250HZ = 25


def reference_r_peaks(annotation, sample_scale=1.0):
    """R-peak sample indices from a wfdb Annotation, marked by the 'N' (QRS) symbol.

    ``sample_scale`` rescales LUDB's 500 Hz annotation indices to the 250 Hz
    processing rate (``sample_scale=0.5``), matching ``annotations.generate_labels``.
    """
    return np.asarray([
        int(round(sample * sample_scale))
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol == 'N'
    ], dtype=int)


def match_r_peaks(detected, reference, tolerance=MATCH_TOLERANCE_SAMPLES_100MS_AT_250HZ):
    """Greedy nearest-neighbor one-to-one matching within `tolerance` samples.

    Returns (matched_pairs, false_positive_indices, false_negative_indices,
    signed_timing_errors_samples) where a signed error is
    ``detected - reference`` for each matched pair (positive = detected late).
    """
    unmatched_ref = list(range(len(reference)))
    matched_pairs = []
    false_positives = []
    for d_idx, d_sample in enumerate(detected):
        best_j, best_dist = None, None
        for j in unmatched_ref:
            dist = abs(int(reference[j]) - int(d_sample))
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_j, best_dist = j, dist
        if best_j is not None:
            matched_pairs.append((d_idx, best_j))
            unmatched_ref.remove(best_j)
        else:
            false_positives.append(d_idx)
    false_negatives = unmatched_ref
    timing_errors = np.asarray([
        int(detected[d_idx]) - int(reference[r_idx]) for d_idx, r_idx in matched_pairs
    ], dtype=int)
    return matched_pairs, false_positives, false_negatives, timing_errors
