#!/usr/bin/env python3
"""Train one ablation configuration (A0-A4 or a developmental R-config) from its YAML file.

Usage:
    python scripts/train_ablation.py --config configs/A1.yaml --seed 1 \
        --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data

A0 uses fixed, non-R-centered sliding windows (src/preprocessing/windowing.py
create_fixed_windows) and is trained/evaluated end-to-end on that scheme; A1-A3
use R-centered windows. A4 does NOT train on QTDB at all: per the paper
("A4 uses the corresponding best A3 checkpoint for each seed as initialization"),
it loads the matching A3 checkpoint (and A3's QTDB normalization stats, saved
alongside it) for the same seed and only runs the 8-epoch LUDB adaptation step.
Run A3 for a given seed before running A4 for that seed.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_qtdb import qtdb_record, qtdb_record_fixed  # noqa: E402
from src.preprocessing.load_ludb import ludb_record  # noqa: E402
from src.preprocessing.windowing import build_partition, build_partition_fixed, time_channel  # noqa: E402
from src.preprocessing.normalization import fit, apply  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2, RPeakTimeML2  # noqa: E402
from src.models.losses import FocalLoss, UnweightedFocalLoss  # noqa: E402
from src.training.train import train_model, set_global_seed  # noqa: E402
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


def load_qtdb_partition(cfg, records, qtdb_dir):
    """Dispatch to fixed-window (A0) or R-centered windowing per the config."""
    if cfg['window'].get('r_centered', True):
        post = cfg['window']['post_r']
        X, y, ids, skipped = build_partition(records, qtdb_record, post, qtdb_dir)
    else:
        length, stride = cfg['window']['length'], cfg['window']['stride']
        X, y, ids, skipped = build_partition_fixed(records, qtdb_record_fixed, length, stride, qtdb_dir)
    return X, y, ids, skipped


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

    post = cfg['window'].get('post_r')  # only meaningful when r_centered
    use_time_channel = cfg['window'].get('temporal_position_channel', False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = MODELS[cfg['model']['architecture'].split(' ')[0]]

    # Seed BEFORE constructing the model so the seed controls weight
    # initialization, not just data-loader ordering (train_model() re-seeds
    # again before building loaders, which is fine/idempotent).
    set_global_seed(args.seed)
    model = model_cls().to(device)

    criterion = build_loss(cfg)

    # ---- A4: load the A3 checkpoint for this seed, adapt on LUDB, done. ----
    if cfg['experiment'] == 'A4':
        base_name = cfg['training']['base_checkpoint']  # 'A3'
        base_ckpt = args.checkpoint_dir / f'{base_name}_seed{args.seed}.pth'
        norm_path = args.checkpoint_dir / f'{base_name}_seed{args.seed}_norm.json'
        if not base_ckpt.exists() or not norm_path.exists():
            raise FileNotFoundError(
                f'Missing {base_name} checkpoint/normalization for seed {args.seed} '
                f'in {args.checkpoint_dir}. Train {base_name} for this seed first.'
            )
        model.load_state_dict(torch.load(base_ckpt, map_location=device))
        norm = json.loads(norm_path.read_text())
        mean, std = norm['mean'], norm['std']

        ludb_adapt = read_split('ludb_adaptation.txt')
        X_adapt, y_adapt, _, _ = build_partition(ludb_adapt, ludb_record, post, args.ludb_dir)
        X_adapt = apply(X_adapt, mean, std)
        X_adapt = time_channel(X_adapt, pre=120, post=post) if use_time_channel else X_adapt[:, None]

        checkpoint_path = args.checkpoint_dir / f'A4_seed{args.seed}.pth'
        adapt_model(model, X_adapt, y_adapt, criterion, args.seed, device, checkpoint_path=checkpoint_path)
        print(f'A4 (adapted from {base_name}, seed {args.seed}) checkpoint saved to {checkpoint_path}')
        return

    # ---- A0-A3 / developmental configs: normal QTDB train + validate. ----
    qtdb_train = read_split('qtdb_train.txt')
    qtdb_val = read_split('qtdb_validation.txt')

    X_train, y_train, _, _ = load_qtdb_partition(cfg, qtdb_train, args.qtdb_dir)
    X_val, y_val, _, _ = load_qtdb_partition(cfg, qtdb_val, args.qtdb_dir)
    mean, std = fit(X_train)
    X_train, X_val = apply(X_train, mean, std), apply(X_val, mean, std)

    if use_time_channel:
        X_train = time_channel(X_train, pre=120, post=post)
        X_val = time_channel(X_val, pre=120, post=post)
    else:
        X_train, X_val = X_train[:, None], X_val[:, None]

    checkpoint_path = args.checkpoint_dir / f"{cfg['experiment']}_seed{args.seed}.pth"
    model, info = train_model(
        model, X_train, y_train, X_val, y_val, criterion, args.seed,
        cfg['training']['epochs'], device, lr=cfg['training']['learning_rate'],
        checkpoint_path=checkpoint_path,
    )
    print(f"Best epoch {info['best_epoch']}, val macro F1 {info['best_validation_macro_f1']:.4f}, "
          f"val loss {info['best_validation_loss']:.4f}")

    # Persist normalization stats so a later A4 run (for the same seed) can
    # reuse exactly this checkpoint's normalization without retraining.
    norm_path = args.checkpoint_dir / f"{cfg['experiment']}_seed{args.seed}_norm.json"
    norm_path.write_text(json.dumps({'mean': mean, 'std': std}))

    print(f'Checkpoint saved to {checkpoint_path}')


if __name__ == '__main__':
    main()
