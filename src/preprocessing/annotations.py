"""Annotation parsing shared by QTDB and LUDB loaders.

Converts wfdb wave-boundary annotations (``(``, ``p``, ``N``, ``t``, ``)``) into
per-sample 3-class labels: ``0`` = Background, ``1`` = P wave, ``2`` = T wave.

QRS-labeled samples (annotation symbol ``N`` for the QRS complex) are mapped to
Background, matching the three-class delineation formulation used throughout the
paper (Section 3.1): the model learns ``f_theta(x) -> {Background, P, T}``.
"""
import numpy as np


def generate_labels(annotation, length, sample_scale=1.0):
    """Build a dense per-sample label array from a wfdb Annotation object.

    Parameters
    ----------
    annotation : wfdb.Annotation
        Annotation object with parallel ``sample`` and ``symbol`` arrays.
    length : int
        Number of samples in the target signal (post-resampling, if any).
    sample_scale : float
        Multiplier applied to each annotation sample index before use. LUDB
        annotations are recorded at 500 Hz and the ECG is resampled to 250 Hz,
        so LUDB callers pass ``sample_scale=0.5``.
    """
    labels = np.zeros(length, dtype=np.int64)
    start = None
    wave_kind = None
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        sample = int(round(sample * sample_scale))
        if symbol == '(':
            start = sample
            wave_kind = None
        elif symbol == 'p':
            wave_kind = 1
        elif symbol == 'N':
            wave_kind = 2  # QRS, later folded into Background
        elif symbol == 't':
            wave_kind = 3
        elif symbol == ')' and start is not None and wave_kind is not None:
            labels[max(0, start): min(length, sample + 1)] = wave_kind
            start = None
            wave_kind = None
    labels[labels == 2] = 0  # QRS -> Background
    labels[labels == 3] = 2  # T -> class index 2
    return labels
