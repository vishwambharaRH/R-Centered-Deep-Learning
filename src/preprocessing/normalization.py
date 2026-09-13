"""Normalization statistics: derived from QTDB training windows only.

Per the paper (Section 3.2.1): "Normalization parameters were derived from the
QTDB training records only and subsequently applied to the QTDB validation and
LUDB data." A separate mean/std pair is computed for the POST=320 (extended
context) window configuration, since its windows differ in length/content.
"""
import numpy as np


def fit(train_windows):
    mean = float(train_windows.mean())
    std = float(train_windows.std())
    if std == 0:
        raise ValueError('Training std is zero; cannot normalize.')
    return mean, std


def apply(windows, mean, std):
    return (windows - mean) / std
