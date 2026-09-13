"""QTDB record loading: native 250 Hz, first recorded channel, pu0 annotations."""
import numpy as np
import wfdb

from .annotations import generate_labels
from .windowing import create_windows
from ..r_peak.pan_tompkins import pan_tompkins_r_peaks


def qtdb_record(record_name, post, qtdb_dir):
    """Load one QTDB record and return R-centered ``(windows, labels)`` arrays."""
    path = str(qtdb_dir / record_name)
    record = wfdb.rdrecord(path)
    ecg = record.p_signal[:, 0].astype(np.float32)
    labels = generate_labels(wfdb.rdann(path, 'pu0'), len(ecg))
    r_peaks = pan_tompkins_r_peaks(ecg, float(record.fs))
    return create_windows(ecg, labels, r_peaks, post)


def list_qtdb_records(qtdb_dir):
    headers = {p.stem for p in qtdb_dir.glob('*.hea')}
    signals = {p.stem for p in qtdb_dir.glob('*.dat')}
    records = sorted(headers & signals)
    if len(records) != 105:
        raise ValueError(f'Expected 105 QTDB records, found {len(records)} in {qtdb_dir}')
    return records
