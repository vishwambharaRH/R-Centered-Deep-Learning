"""Pan-Tompkins R-peak detector used as the physiological temporal reference.

This is the deterministic detector described in Section 3.4 of the paper and
independently audited in Section 4.6 / Table 6 (QTDB: sensitivity 0.9780, PPV
0.9863, F1 0.9804, mean timing error 13.31 ms; LUDB: sensitivity 0.9753, PPV
0.7971, F1 0.8800, mean timing error 3.07 ms).
"""
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def pan_tompkins_r_peaks(ecg, fs):
    """Detect R-peak sample indices in a single-lead ECG signal.

    Steps: bandpass filter (5-18 Hz, 3rd-order Butterworth, zero-phase) ->
    derivative -> squaring -> moving-window integration (150 ms window) ->
    adaptive SPKI/NPKI peak thresholding -> local-maximum refinement within
    a 100 ms search window around each accepted peak.
    """
    nyquist = fs / 2.0
    sos = butter(3, [5.0 / nyquist, 18.0 / nyquist], btype='bandpass', output='sos')
    bandpassed = sosfiltfilt(sos, ecg)
    derivative = np.convolve(bandpassed, np.array([-1, -2, 0, 2, 1]) * fs / 8.0, mode='same')
    width = max(1, round(0.150 * fs))
    integrated = np.convolve(derivative ** 2, np.ones(width) / width, mode='same')
    candidates, _ = find_peaks(integrated, distance=max(1, round(0.20 * fs)))
    boot = candidates[candidates < min(len(ecg), round(2 * fs))]
    spki = np.percentile(integrated[boot], 90) if len(boot) else 0.0
    npki = np.percentile(integrated[boot], 25) if len(boot) else 0.0
    accepted = []
    for peak in candidates:
        threshold = npki + 0.25 * (spki - npki)
        if integrated[peak] >= threshold:
            accepted.append(peak)
            spki = 0.125 * integrated[peak] + 0.875 * spki
        else:
            npki = 0.125 * integrated[peak] + 0.875 * npki
    search = round(0.10 * fs)
    refined = []
    for peak in accepted:
        left = max(0, peak - search)
        right = min(len(ecg), peak + search + 1)
        refined.append(left + int(np.argmax(ecg[left:right])))
    return np.unique(np.asarray(refined, dtype=int))
