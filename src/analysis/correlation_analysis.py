"""R-localization error vs. downstream P/T delineation performance (paper Section 4.7, Table 7).

Computes, per LUDB test record: R-error measures (total errors, missed-R rate,
false-positive-R rate, mean R timing error) against a Pan-Tompkins detector
audit, and downstream performance (P-wave F1, T-wave F1, P/T Macro F1).
Spearman rank correlation is then computed across records between each error
measure and each downstream performance measure, with two-sided significance.
"""
import numpy as np
from scipy.stats import spearmanr


def r_error_measures(matched_r_peaks, missed_r, false_positive_r, timing_errors_ms):
    """Summarize one record's R-peak audit outcome.

    `matched_r_peaks`, `missed_r`, `false_positive_r` are counts;
    `timing_errors_ms` is an array of signed timing errors for matched peaks.
    """
    total_errors = missed_r + false_positive_r
    total_r = matched_r_peaks + missed_r
    return {
        'total_r_errors': total_errors,
        'missed_r_rate': missed_r / total_r if total_r else float('nan'),
        'false_positive_r_rate': false_positive_r / total_r if total_r else float('nan'),
        'r_timing_error_ms': float(np.mean(np.abs(timing_errors_ms))) if len(timing_errors_ms) else float('nan'),
    }


def spearman_correlation_table(records_df, error_columns, outcome_columns):
    """records_df: one row per record with error_columns + outcome_columns.

    Returns a nested dict: {error_col: {outcome_col: (rho, p_value)}}.
    """
    results = {}
    for err_col in error_columns:
        results[err_col] = {}
        for out_col in outcome_columns:
            mask = records_df[[err_col, out_col]].notna().all(axis=1)
            rho, p_value = spearmanr(records_df.loc[mask, err_col], records_df.loc[mask, out_col])
            results[err_col][out_col] = (rho, p_value)
    return results


def significance_code(p_value):
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''
