"""R-centered window construction.

Windows are anchored on Pan-Tompkins-detected R peaks (``src/r_peak/pan_tompkins.py``),
not on annotated QRS positions, so the same detector drives both the model input
representation and the independent R-peak audit in Section 4.6 of the paper.
"""
import numpy as np

PRE = 120     # samples preceding the R peak (fixed across all configurations)
POST = 240    # primary post-R context -> 360-sample windows (A1/R1-R3)
POST_EXTENDED = 320  # extended post-R context -> 440-sample windows (A2+/R5-R6)


def create_windows(ecg, labels, r_peaks, post):
    """Slice ``[r - PRE, r + post)`` windows around each detected R peak.

    A window is dropped if it would run off either end of the record, so the
    number of windows per record is at most ``len(r_peaks)`` and can be lower
    near the record boundaries.
    """
    xs, ys = [], []
    for r in r_peaks:
        left, right = r - PRE, r + post
        if left >= 0 and right <= len(ecg):
            xs.append(ecg[left:right])
            ys.append(labels[left:right])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def build_partition(record_names, builder, post, data_dir):
    """Concatenate windows across a list of records, tracking per-window record ids.

    ``builder`` is one of ``load_qtdb.qtdb_record`` / ``load_ludb.ludb_record``.
    Records that raise during loading are skipped and reported rather than
    failing the whole partition, matching the behavior used to produce the
    paper's reported window counts.
    """
    xs, ys, ids = [], [], []
    skipped = []
    for record_name in record_names:
        try:
            x, y = builder(record_name, post, data_dir)
            if len(x):
                xs.append(x)
                ys.append(y)
                ids.extend([record_name] * len(x))
        except Exception as exc:  # pragma: no cover - data-specific guard
            skipped.append({'record': record_name, 'error': str(exc)})
    if not xs:
        raise RuntimeError('No usable windows generated for the selected partition.')
    return np.concatenate(xs), np.concatenate(ys), np.asarray(ids), skipped


def time_channel(signals, pre, post):
    """R-relative normalized time channel used by the A3/R6 temporal-position variant.

    Ranges from ``-PRE/post`` at the window's left edge, through ``0`` at the R
    peak, to ``+1`` at the window's right edge.
    """
    time = (np.arange(signals.shape[1], dtype=np.float32) - pre) / float(post)
    stacked = np.stack([signals.astype(np.float32), np.broadcast_to(time, signals.shape)], axis=1)
    return stacked.copy()
