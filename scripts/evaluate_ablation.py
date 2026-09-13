#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the 180-record LUDB held-out test set.

Usage:
    python scripts/evaluate_ablation.py --config configs/A1.yaml \
        --checkpoint checkpoints/A1_seed1.pth --ludb-dir /path/to/ludb/1.0.1/data

Normalization stats are read from the ``<experiment>_seed<seed>_norm.json``
file written alongside the checkpoint by train_ablation.py, unless
--train-mean/--train-std are passed explicitly. A4 checkpoints reuse A3's
normalization file (see configs/A4.yaml's ``base_checkpoint``).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_ludb import ludb_record, ludb_record_fixed  # noqa: E402
from src.preprocessing.windowing import build_partition, build_partition_fixed, time_channel  # noqa: E402
from src.preprocessing.normalization import apply  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2, RPeakTimeML2  # noqa: E402
from src.training.train import predict  # noqa: E402
from src.training.evaluate import full_report  # noqa: E402

MODELS = {'RPeakGuidedML2': RPeakGuidedML2, 'RPeakTimeML2': RPeakTimeML2}


def read_split(name):
    return [l.strip() for l in (ROOT / 'data' / 'splits' / name).read_text().splitlines() if l.strip()]


def resolve_norm_path(checkpoint_path, cfg):
    """A4 checkpoints (A4_seed<N>.pth) reuse A3_seed<N>_norm.json; others use their own."""
    match = re.match(r'(.+)_seed(\d+)\.pth$', checkpoint_path.name)
    if not match:
        return None
    experiment, seed = match.group(1), match.group(2)
    if experiment == 'A4':
        experiment = cfg['training'].get('base_checkpoint', 'A3')
    return checkpoint_path.parent / f'{experiment}_seed{seed}_norm.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--train-mean', type=float, default=None)
    parser.add_argument('--train-std', type=float, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    r_centered = cfg['window'].get('r_centered', True)
    post = cfg['window'].get('post_r')
    use_time_channel = cfg['window'].get('temporal_position_channel', False)

    if args.train_mean is not None and args.train_std is not None:
        mean, std = args.train_mean, args.train_std
    else:
        norm_path = resolve_norm_path(args.checkpoint, cfg)
        if norm_path is None or not norm_path.exists():
            raise FileNotFoundError(
                'Could not find a normalization JSON next to the checkpoint; '
                'pass --train-mean/--train-std explicitly.'
            )
        norm = json.loads(norm_path.read_text())
        mean, std = norm['mean'], norm['std']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = MODELS[cfg['model']['architecture'].split(' ')[0]]
    model = model_cls().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    ludb_test = read_split('ludb_test.txt')
    if r_centered:
        X_test, y_test, _, _ = build_partition(ludb_test, ludb_record, post, args.ludb_dir)
    else:
        length, stride = cfg['window']['length'], cfg['window']['stride']
        X_test, y_test, _, _ = build_partition_fixed(ludb_test, ludb_record_fixed, length, stride, args.ludb_dir)

    X_test = apply(X_test, mean, std)
    X_test = time_channel(X_test, pre=120, post=post) if use_time_channel else X_test[:, None]

    y_pred = predict(model, X_test, device)
    report = full_report(y_test, y_pred)
    print(f"Macro F1: {report['macro_f1']:.4f}  Weighted F1: {report['weighted_f1']:.4f}  "
          f"Accuracy: {report['accuracy']:.4f}")


if __name__ == '__main__':
    main()
