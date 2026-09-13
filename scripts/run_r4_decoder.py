#!/usr/bin/env python3
"""Reproduce the R4 failure-analysis experiment end to end.

Loads a frozen R1 checkpoint (configs/developmental/R1.yaml), locks the
T-probability threshold on the QTDB validation set, then applies the decoder
(src/analysis/r4_decoder.py) to QTDB validation and the LUDB test set.

Usage:
    python scripts/run_r4_decoder.py --r1-checkpoint checkpoints/R1_seed1.pth \
        --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data \
        --train-mean <float> --train-std <float>
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_qtdb import qtdb_record  # noqa: E402
from src.preprocessing.load_ludb import ludb_record  # noqa: E402
from src.preprocessing.windowing import build_partition  # noqa: E402
from src.preprocessing.normalization import apply  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2  # noqa: E402
from src.training.evaluate import macro_f1  # noqa: E402
from src.analysis.r4_decoder import softmax_probabilities, select_t_threshold, decode  # noqa: E402

POST = 240  # R1 uses the primary 360-sample (PRE=120, POST=240) window


def read_split(name):
    return [l.strip() for l in (ROOT / 'data' / 'splits' / name).read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1-checkpoint', required=True, type=Path)
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--train-mean', required=True, type=float)
    parser.add_argument('--train-std', required=True, type=float)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RPeakGuidedML2().to(device)
    model.load_state_dict(torch.load(args.r1_checkpoint, map_location=device))

    qtdb_val = read_split('qtdb_validation.txt')
    ludb_test = read_split('ludb_test.txt')

    X_val, y_val, _, _ = build_partition(qtdb_val, qtdb_record, POST, args.qtdb_dir)
    X_val = apply(X_val, args.train_mean, args.train_std)[:, None]
    probs_val = softmax_probabilities(model, X_val, device)

    threshold, val_t_f1 = select_t_threshold(probs_val, y_val)
    print(f'Locked T-probability threshold: {threshold:.2f} (QTDB val T-class F1 at lock time: {val_t_f1:.4f})')

    val_decoded = decode(probs_val, threshold)
    print(f'QTDB validation Macro F1 after R-aware decoding: {macro_f1(y_val, val_decoded):.4f}')

    X_test, y_test, _, _ = build_partition(ludb_test, ludb_record, POST, args.ludb_dir)
    X_test = apply(X_test, args.train_mean, args.train_std)[:, None]
    probs_test = softmax_probabilities(model, X_test, device)
    test_decoded = decode(probs_test, threshold)
    print(f'LUDB test Macro F1 after R-aware decoding: {macro_f1(y_test, test_decoded):.4f}')
    print('Reference (paper, Table 3): QTDB 0.7328, LUDB 0.7119 — see configs/developmental/R4.yaml')


if __name__ == '__main__':
    main()
