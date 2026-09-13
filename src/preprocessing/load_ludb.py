"""LUDB record loading: resampled 500 -> 250 Hz, lead II, ii annotations."""
import numpy as np
from scipy.signal import resample_poly
import wfdb

from .annotations import generate_labels
from .windowing import create_windows, create_fixed_windows
from ..r_peak.pan_tompkins import pan_tompkins_r_peaks


def ludb_record(record_name, post, ludb_dir):
    """Load one LUDB record and return R-centered ``(windows, labels)`` arrays.

    LUDB's lead II is used as the counterpart to QTDB's MLII channel. The
    signal is polyphase-resampled from 500 Hz to 250 Hz (``up=1, down=2``) and
    annotation sample indices are scaled by the same factor (``sample_scale=0.5``).
    """
    path = str(ludb_dir / record_name)
    record = wfdb.rdrecord(path)
    lead = record.sig_name.index('ii')
    ecg = resample_poly(record.p_signal[:, lead].astype(np.float32), up=1, down=2)
    labels = generate_labels(wfdb.rdann(path, 'ii'), len(ecg), sample_scale=0.5)
    size = min(len(ecg), len(labels))
    ecg, labels = ecg[:size], labels[:size]
    r_peaks = pan_tompkins_r_peaks(ecg, 250.0)
    return create_windows(ecg, labels, r_peaks, post)


def ludb_record_fixed(record_name, length, stride, ludb_dir):
    """Load one LUDB record and return fixed-stride, non-R-centered windows (A0 baseline)."""
    path = str(ludb_dir / record_name)
    record = wfdb.rdrecord(path)
    lead = record.sig_name.index('ii')
    ecg = resample_poly(record.p_signal[:, lead].astype(np.float32), up=1, down=2)
    labels = generate_labels(wfdb.rdann(path, 'ii'), len(ecg), sample_scale=0.5)
    size = min(len(ecg), len(labels))
    ecg, labels = ecg[:size], labels[:size]
    return create_fixed_windows(ecg, labels, length, stride)


def list_ludb_records(ludb_dir):
    headers = {p.stem for p in ludb_dir.glob('*.hea')}
    signals = {p.stem for p in ludb_dir.glob('*.dat')}
    records = sorted(headers & signals)
    if len(records) != 200:
        raise ValueError(f'Expected 200 LUDB records, found {len(records)} in {ludb_dir}')
    return records
