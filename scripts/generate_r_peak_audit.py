#!/usr/bin/env python3
"""Regenerate results/r_peak/r_peak_audit.csv (Table 6) from raw QTDB/LUDB data.

Runs the Pan-Tompkins detector (src/r_peak/pan_tompkins.py) against every
record in each database and matches its output one-to-one against the 'N'
(QRS) reference annotation, independent of any downstream P/T model.

By default this audits every record in each database (105 QTDB, 200 LUDB);
pass --qtdb-records-file / --ludb-records-file to restrict to a specific
split (e.g. data/splits/qtdb_validation.txt) if you are trying to match a
particular reported subset.

Usage:
    python scripts/generate_r_peak_audit.py --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_qtdb import list_qtdb_records  # noqa: E402
from src.preprocessing.load_ludb import list_ludb_records  # noqa: E402
from src.r_peak.pan_tompkins import pan_tompkins_r_peaks  # noqa: E402
from src.r_peak.matching import reference_r_peaks  # noqa: E402
from src.r_peak.audit import audit_record, aggregate_audit  # noqa: E402


def read_records_file(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def audit_qtdb(records, qtdb_dir):
    rows = []
    for record_name in records:
        path = str(qtdb_dir / record_name)
        record = wfdb.rdrecord(path)
        ecg = record.p_signal[:, 0].astype(np.float32)
        detected = pan_tompkins_r_peaks(ecg, float(record.fs))
        reference = reference_r_peaks(wfdb.rdann(path, 'pu0'))
        rows.append({'record': record_name, **audit_record(detected, reference, sampling_rate_hz=record.fs)})
    return rows


def audit_ludb(records, ludb_dir):
    rows = []
    for record_name in records:
        path = str(ludb_dir / record_name)
        record = wfdb.rdrecord(path)
        lead = record.sig_name.index('ii')
        ecg = resample_poly(record.p_signal[:, lead].astype(np.float32), up=1, down=2)
        detected = pan_tompkins_r_peaks(ecg, 250.0)
        reference = reference_r_peaks(wfdb.rdann(path, 'ii'), sample_scale=0.5)
        rows.append({'record': record_name, **audit_record(detected, reference, sampling_rate_hz=250.0)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--qtdb-records-file', type=Path, default=None)
    parser.add_argument('--ludb-records-file', type=Path, default=None)
    args = parser.parse_args()

    qtdb_records = read_records_file(args.qtdb_records_file) if args.qtdb_records_file else list_qtdb_records(args.qtdb_dir)
    ludb_records = read_records_file(args.ludb_records_file) if args.ludb_records_file else list_ludb_records(args.ludb_dir)

    print(f'Auditing {len(qtdb_records)} QTDB records...')
    qtdb_rows = audit_qtdb(qtdb_records, args.qtdb_dir)
    print(f'Auditing {len(ludb_records)} LUDB records...')
    ludb_rows = audit_ludb(ludb_records, args.ludb_dir)

    out_dir = ROOT / 'results' / 'r_peak'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qtdb_rows).to_csv(out_dir / 'r_peak_audit_qtdb_per_record.csv', index=False)
    pd.DataFrame(ludb_rows).to_csv(out_dir / 'r_peak_audit_ludb_per_record.csv', index=False)

    qtdb_agg = aggregate_audit(qtdb_rows)
    ludb_agg = aggregate_audit(ludb_rows)
    summary = pd.DataFrame([
        {'dataset': 'QTDB', **qtdb_agg},
        {'dataset': 'LUDB', **ludb_agg},
    ])
    summary.to_csv(out_dir / 'r_peak_audit.csv', index=False)
    print(summary.to_string(index=False))
    print(f'\nWrote {out_dir}/r_peak_audit.csv (compare against the committed reference, Table 6)')


if __name__ == '__main__':
    main()
