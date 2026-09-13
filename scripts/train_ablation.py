#!/usr/bin/env python3
"""Train one ablation configuration (A0-A4 or a developmental R-config) from its YAML file.

Usage:
    python scripts/train_ablation.py --config configs/A1.yaml --seed 1 \
        --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_qtdb import qtdb_record  # noqa: E402
from src.preprocessing.load_ludb import ludb_record  # noqa: E402
from src.preprocessing.windowing import build_partition, time_channel  # noqa: E402
from src.preprocessing.normalization import fit, apply  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2, RPeakTimeML2  # noqa: E402
from src.models.losses import FocalLoss, UnweightedFocalLoss  # noqa: E402
from src.training.train import train_model  # noqa: E402
from src.training.adapt import adapt_model  # noqa: E402

MODELS = {'RPeakGuidedML2': RPeakGuidedML2, 'RPeakTimeML2': RPeakTimeML2}


def read_split(name):
    return [l.strip() for l in (ROOT / 'data' / 'splits' / name).read_text().splitlines() if l.strip()]


def build_loss(cfg):
    loss_type = cfg['loss']['type']
    gamma = cfg['loss'].get('gamma', 2.0)
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if loss_type == 'unweighted_focal':
        return UnweightedFocalLoss(gamma=gamma)
    if loss_type == 'weighted_focal':
        raise NotImplementedError('Compute class weights from the training partition and pass as alpha.')
    raise ValueError(f'Unknown loss type: {loss_type}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--checkpoint-dir', default=ROOT / 'checkpoints', type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    post = cfg['window']['post_r'] if cfg['window'].get('r_centered', True) else cfg['window']['length'] - 120
    use_time_channel = cfg['window'].get('temporal_position_channel', False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = MODELS[cfg['model']['architecture'].split(' ')[0]]
    model = model_cls().to(device)

    qtdb_train = read_split('qtdb_train.txt')
    qtdb_val = read_split('qtdb_validation.txt')

    X_train, y_train, _, _ = build_partition(qtdb_train, qtdb_record, post, args.qtdb_dir)
    X_val, y_val, _, _ = build_partition(qtdb_val, qtdb_record, post, args.qtdb_dir)
    mean, std = fit(X_train)
    X_train, X_val = apply(X_train, mean, std), apply(X_val, mean, std)

    if use_time_channel:
        X_train = time_channel(X_train, pre=120, post=post)
        X_val = time_channel(X_val, pre=120, post=post)
    else:
        X_train, X_val = X_train[:, None], X_val[:, None]

    criterion = build_loss(cfg)
    checkpoint_path = args.checkpoint_dir / f"{cfg['experiment']}_seed{args.seed}.pth"
    model, info = train_model(
        model, X_train, y_train, X_val, y_val, criterion, args.seed,
        cfg['training']['epochs'], device, lr=cfg['training']['learning_rate'],
        checkpoint_path=checkpoint_path,
    )
    print(f"Best epoch {info['best_epoch']}, val loss {info['best_validation_loss']:.4f}")

    if cfg['experiment'] == 'A4':
        ludb_adapt = read_split('ludb_adaptation.txt')
        X_adapt, y_adapt, _, _ = build_partition(ludb_adapt, ludb_record, post, args.ludb_dir)
        X_adapt = apply(X_adapt, mean, std)
        X_adapt = time_channel(X_adapt, pre=120, post=post) if use_time_channel else X_adapt[:, None]
        adapt_model(model, X_adapt, y_adapt, criterion, args.seed, device,
                    checkpoint_path=args.checkpoint_dir / f"A4_adapted_seed{args.seed}.pth")

    print(f'Checkpoint saved to {checkpoint_path}')


if __name__ == '__main__':
    main()
