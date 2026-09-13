#!/usr/bin/env python3
"""Regenerate results/correlations/r_error_correlations.csv (Table 7).

For every LUDB test record: (1) run the Pan-Tompkins R-peak audit against
that record's reference R annotations to get R-error measures, and (2) run
the trained model on that record's windows to get downstream P-wave F1,
T-wave F1, and P/T Macro F1. Then Spearman-correlate the R-error measures
against the downstream measures across all 180 records.

Requires a trained, adapted checkpoint (e.g. an A4 checkpoint) — pass its
config and normalization stats the same way as scripts/evaluate_ablation.py.

Usage:
    python scripts/generate_correlation_analysis.py --config configs/A4.yaml \
        --checkpoint checkpoints/A4_seed1.pth --ludb-dir /path/to/ludb/1.0.1/data
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.annotations import generate_labels  # noqa: E402
from src.preprocessing.windowing import create_windows, time_channel  # noqa: E402
from src.preprocessing.normalization import apply  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2, RPeakTimeML2  # noqa: E402
from src.training.train import predict  # noqa: E402
from src.r_peak.pan_tompkins import pan_tompkins_r_peaks  # noqa: E402
from src.r_peak.matching import reference_r_peaks  # noqa: E402
from src.r_peak.audit import audit_record  # noqa: E402
from src.analysis.event_metrics import event_level_metrics  # noqa: E402
from src.analysis.correlation_analysis import spearman_correlation_table, significance_code  # noqa: E402

MODELS = {'RPeakGuidedML2': RPeakGuidedML2, 'RPeakTimeML2': RPeakTimeML2}


def read_split(name):
    return [l.strip() for l in (ROOT / 'data' / 'splits' / name).read_text().splitlines() if l.strip()]


def resolve_norm_path(checkpoint_path, cfg):
    match = re.match(r'(.+)_seed(\d+)\.pth$', checkpoint_path.name)
    experiment, seed = match.group(1), match.group(2)
    if experiment == 'A4':
        experiment = cfg['training'].get('base_checkpoint', 'A3')
    return checkpoint_path.parent / f'{experiment}_seed{seed}_norm.json'


def per_record_analysis(record_name, ludb_dir, model, device, post, use_time_channel, mean, std):
    path = str(ludb_dir / record_name)
    record = wfdb.rdrecord(path)
    lead = record.sig_name.index('ii')
    ecg = resample_poly(record.p_signal[:, lead].astype(np.float32), up=1, down=2)
    annotation = wfdb.rdann(path, 'ii')
    labels = generate_labels(annotation, len(ecg), sample_scale=0.5)
    size = min(len(ecg), len(labels))
    ecg, labels = ecg[:size], labels[:size]

    detected_r = pan_tompkins_r_peaks(ecg, 250.0)
    reference_r = reference_r_peaks(annotation, sample_scale=0.5)
    r_audit = audit_record(detected_r, reference_r, sampling_rate_hz=250.0)
    total_r = r_audit['tp'] + r_audit['fn']
    r_errors = {
        'total_r_errors': r_audit['fp'] + r_audit['fn'],
        'missed_r_rate': r_audit['fn'] / total_r if total_r else np.nan,
        'false_positive_r_rate': r_audit['fp'] / total_r if total_r else np.nan,
        'r_timing_error_ms': r_audit['mean_timing_error_ms'],
    }

    X, y = create_windows(ecg, labels, detected_r, post)
    if not len(X):
        return {**r_errors, 'record': record_name, 'p_f1': np.nan, 't_f1': np.nan, 'pt_macro_f1': np.nan}
    X = apply(X, mean, std)
    X = time_channel(X, pre=120, post=post) if use_time_channel else X[:, None]
    y_pred = predict(model, X, device)

    p_f1 = event_level_metrics(y, y_pred, class_index=1)['f1']
    t_f1 = event_level_metrics(y, y_pred, class_index=2)['f1']
    pt_macro_f1 = np.nanmean([p_f1, t_f1])
    return {**r_errors, 'record': record_name, 'p_f1': p_f1, 't_f1': t_f1, 'pt_macro_f1': pt_macro_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    post = cfg['window'].get('post_r')
    use_time_channel = cfg['window'].get('temporal_position_channel', False)

    norm_path = resolve_norm_path(args.checkpoint, cfg)
    norm = json.loads(norm_path.read_text())
    mean, std = norm['mean'], norm['std']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = MODELS[cfg['model']['architecture'].split(' ')[0]]
    model = model_cls().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    ludb_test = read_split('ludb_test.txt')
    rows = [per_record_analysis(r, args.ludb_dir, model, device, post, use_time_channel, mean, std)
            for r in ludb_test]
    df = pd.DataFrame(rows)

    error_cols = ['total_r_errors', 'missed_r_rate', 'false_positive_r_rate', 'r_timing_error_ms']
    outcome_cols = ['p_f1', 't_f1', 'pt_macro_f1']
    table = spearman_correlation_table(df, error_cols, outcome_cols)

    out_rows = []
    for err_col in error_cols:
        row = {'r_error_measure': err_col}
        for out_col in outcome_cols:
            rho, p_value = table[err_col][out_col]
            row[f'{out_col}_spearman_rho'] = round(rho, 4)
            row[f'{out_col}_sig'] = significance_code(p_value)
        out_rows.append(row)

    out_dir = ROOT / 'results' / 'correlations'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'per_record_r_error_and_f1.csv', index=False)
    pd.DataFrame(out_rows).to_csv(out_dir / 'r_error_correlations.csv', index=False)
    print(pd.DataFrame(out_rows).to_string(index=False))
    print(f'\nWrote {out_dir}/r_error_correlations.csv (compare against the committed reference, Table 7)')


if __name__ == '__main__':
    main()
