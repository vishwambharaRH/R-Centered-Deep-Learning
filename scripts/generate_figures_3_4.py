#!/usr/bin/env python3
"""Regenerate Figures 3 and 4 with the corrections from misc/changesToFig3and4.txt.

Uses the exact same representative examples as the published figures
(QTDB_IDX=15, LUDB_IDX=20 — the notebook cell these were produced from is
PAPER/experimentalModel2.ipynb, cells 80-81) and the exact same trained
checkpoints, so the underlying signal/prediction shown is unchanged. Only the
presentation changes:

  - An explicit Background / P / T legend is added to every mask panel.
    QRS is never shown as its own class, since the model only predicts
    {Background, P, T} (Section 3.1) — QRS-labeled samples are folded into
    Background before training, so it has no color slot at all here (the
    published figure's colormap carried an unused fourth "QRS" color that
    could be misread as a fourth predicted class; it's removed).
  - Panel titles say what classes their colors represent, e.g.
    "Ground Truth (Background / P / T)".
  - Fig. 3's caption no longer calls a multi-cycle segment "a beat" —
    both captions now say "representative ... ECG segment".

Usage:
    python scripts/generate_figures_3_4.py \
        --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data \
        --r3-checkpoint /path/to/best_ml2_rpeak_guided_focal_unweighted.pth \
        --r6-checkpoint /path/to/best_ml2_r6_time_ludb10.pth \
        --normalization /path/to/best_ml2_rpeak_guided_normalization.npz
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_qtdb import qtdb_record, list_qtdb_records  # noqa: E402
from src.preprocessing.load_ludb import ludb_record, list_ludb_records  # noqa: E402
from src.preprocessing.windowing import build_partition, time_channel  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2, RPeakTimeML2  # noqa: E402
from src.training.train import predict  # noqa: E402 — batches inference (BATCH_SIZE=64)

QTDB_IDX = 15
LUDB_IDX = 20
PRE, POST = 120, 320  # R5/R6 window used for both figures in the source notebook

# Background is not drawn as a filled color at all (white/transparent), so the
# legend and colormap only ever need two data colors: P and T.
COLOR_P = '#66c2a5'
COLOR_T = '#fc8d62'
CMAP = ListedColormap(['#ffffff', COLOR_P, COLOR_T])
LEGEND_HANDLES = [
    Patch(facecolor='#ffffff', edgecolor='#888888', label='Background'),
    Patch(facecolor=COLOR_P, label='P wave'),
    Patch(facecolor=COLOR_T, label='T wave'),
]


def draw_mask(ax, labels, title):
    labels = np.asarray(labels).astype(int)
    ax.imshow(labels[np.newaxis, :], cmap=CMAP, aspect='auto', interpolation='nearest',
              vmin=0, vmax=2, extent=[0, len(labels), 0, 1])
    ax.set_xlim(0, len(labels))
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--r3-checkpoint', required=True, type=Path)
    parser.add_argument('--r6-checkpoint', required=True, type=Path)
    parser.add_argument('--normalization', required=True, type=Path,
                         help='.npz with train_mean/train_std (R1/R3 normalization)')
    parser.add_argument('--out-dir', default=ROOT / 'figures', type=Path)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- QTDB: build R5/R6-window (PRE=120, POST=320) beats, split, normalize ----
    qtdb_records = list_qtdb_records(args.qtdb_dir)
    X_r5, Y_r5, ids_r5, _ = build_partition(qtdb_records, qtdb_record, POST, args.qtdb_dir)
    train_records, val_records = train_test_split(np.unique(ids_r5), test_size=0.20, random_state=7, shuffle=True)
    val_mask = np.isin(ids_r5, val_records)

    norm = np.load(args.normalization)
    train_mean, train_std = float(norm['train_mean']), float(norm['train_std'])
    X_val = (X_r5[val_mask] - train_mean) / train_std
    Y_val = Y_r5[val_mask]

    model_r3 = RPeakGuidedML2().to(device)
    model_r3.load_state_dict(torch.load(args.r3_checkpoint, map_location=device))
    model_r3.eval()
    r3_val_pred = predict(model_r3, X_val[:, None], device)

    signal_qtdb = X_val[QTDB_IDX]
    truth_qtdb = Y_val[QTDB_IDX]
    pred_qtdb = r3_val_pred[QTDB_IDX]

    # ---- Figure 3 ----
    fig, ax = plt.subplots(3, 1, figsize=(13, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    ax[0].plot(signal_qtdb, color='black', lw=1.2)
    ax[0].set_title('QTDB ECG Signal', fontsize=13)
    ax[0].set_ylabel('Normalized Amplitude')
    ax[0].grid(alpha=0.25)
    draw_mask(ax[1], truth_qtdb, 'Ground Truth (Background / P / T)')
    draw_mask(ax[2], pred_qtdb, 'R3 Prediction (Background / P / T)')
    ax[2].set_xlabel('Sample Index')
    ax[0].legend(handles=LEGEND_HANDLES, loc='upper right', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Representative QTDB validation ECG segment', fontsize=11, y=1.01)
    plt.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / 'qtdb_example_corrected.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.out_dir / "qtdb_example_corrected.png"}')

    # ---- LUDB: build R5/R6-window beats, R3 baseline + R6 adapted predictions ----
    ludb_records = list_ludb_records(args.ludb_dir)
    X_ludb, Y_ludb, ids_ludb, _ = build_partition(ludb_records, ludb_record, POST, args.ludb_dir)
    adapt_records, test_records = train_test_split(np.unique(ids_ludb), test_size=0.90, random_state=17, shuffle=True)
    test_mask = np.isin(ids_ludb, test_records)

    X_ludb_r3_norm = (X_ludb - train_mean) / train_std
    r3_ludb_pred = predict(model_r3, X_ludb_r3_norm[:, None], device)

    r5_mean, r5_std = float(X_r5[np.isin(ids_r5, train_records)].mean()), float(X_r5[np.isin(ids_r5, train_records)].std())
    X_ludb_test_r6 = time_channel((X_ludb[test_mask] - r5_mean) / r5_std, pre=PRE, post=POST)

    model_r6 = RPeakTimeML2().to(device)
    model_r6.load_state_dict(torch.load(args.r6_checkpoint, map_location=device))
    model_r6.eval()
    r6_pred = predict(model_r6, X_ludb_test_r6, device)

    signal_ludb = X_ludb[test_mask][LUDB_IDX]
    truth_ludb = Y_ludb[test_mask][LUDB_IDX]
    baseline_ludb = r3_ludb_pred[test_mask][LUDB_IDX]
    proposed_ludb = r6_pred[LUDB_IDX]

    # ---- Figure 4 ----
    fig, ax = plt.subplots(4, 1, figsize=(13, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    ax[0].plot(signal_ludb, color='black', lw=1.2)
    ax[0].set_title('LUDB ECG Signal', fontsize=13)
    ax[0].set_ylabel('Normalized Amplitude')
    ax[0].grid(alpha=0.25)
    draw_mask(ax[1], truth_ludb, 'Ground Truth (Background / P / T)')
    draw_mask(ax[2], baseline_ludb, 'Baseline R3 (Background / P / T)')
    draw_mask(ax[3], proposed_ludb, 'Proposed Framework R6 (Background / P / T)')
    ax[3].set_xlabel('Sample Index')
    ax[0].legend(handles=LEGEND_HANDLES, loc='upper right', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Representative LUDB held-out ECG segment', fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(args.out_dir / 'ludb_example_corrected.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.out_dir / "ludb_example_corrected.png"}')


if __name__ == '__main__':
    main()
