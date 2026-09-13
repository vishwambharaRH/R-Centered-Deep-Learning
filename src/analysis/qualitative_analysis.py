"""Per-record qualitative / error analysis (paper Section 4.8).

Identifies difficult records — e.g. those with P-wave F1 of 0 combined with a
high total R-detection error count — and reports the corresponding T-wave F1
to illustrate that poor R localization can coincide with severe P-wave
degradation without proportionally degrading T-wave performance.
"""
import pandas as pd


def flag_difficult_records(per_record_df, p_f1_col='p_f1', t_f1_col='t_f1',
                            r_error_col='total_r_errors', p_f1_threshold=0.0):
    """Return rows where P-wave F1 is at/below `p_f1_threshold`, sorted by R error (desc)."""
    difficult = per_record_df[per_record_df[p_f1_col] <= p_f1_threshold].copy()
    return difficult.sort_values(r_error_col, ascending=False)


def summarize_qualitative_cases(per_record_df, **kwargs):
    difficult = flag_difficult_records(per_record_df, **kwargs)
    return difficult[['record', kwargs.get('p_f1_col', 'p_f1'),
                       kwargs.get('t_f1_col', 't_f1'),
                       kwargs.get('r_error_col', 'total_r_errors')]]
