#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the 180-record LUDB held-out test set.

Usage:
    python scripts/evaluate_ablation.py --config configs/A1.yaml \
        --checkpoint checkpoints/A1_seed1.pth --ludb-dir /path/to/ludb/1.0.1/data
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_ludb import ludb_record  # noqa: E402
from src.preprocessing.windowing import build_partition, time_channel  # noqa: E402
from src.preprocessing.normalization import apply  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2, RPeakTimeML2  # noqa: E402
from src.training.train import predict  # noqa: E402
from src.training.evaluate import full_report  # noqa: E402

MODELS = {'RPeakGuidedML2': RPeakGuidedML2, 'RPeakTimeML2': RPeakTimeML2}


def read_split(name):
    return [l.strip() for l in (ROOT / 'data' / 'splits' / name).read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--train-mean', required=True, type=float)
    parser.add_argument('--train-std', required=True, type=float)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    post = cfg['window']['post_r'] if cfg['window'].get('r_centered', True) else cfg['window']['length'] - 120
    use_time_channel = cfg['window'].get('temporal_position_channel', False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = MODELS[cfg['model']['architecture'].split(' ')[0]]
    model = model_cls().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    ludb_test = read_split('ludb_test.txt')
    X_test, y_test, _, _ = build_partition(ludb_test, ludb_record, post, args.ludb_dir)
    X_test = apply(X_test, args.train_mean, args.train_std)
    X_test = time_channel(X_test, pre=120, post=post) if use_time_channel else X_test[:, None]

    y_pred = predict(model, X_test, device)
    report = full_report(y_test, y_pred)
    print(f"Macro F1: {report['macro_f1']:.4f}  Weighted F1: {report['weighted_f1']:.4f}  "
          f"Accuracy: {report['accuracy']:.4f}")


if __name__ == '__main__':
    main()
