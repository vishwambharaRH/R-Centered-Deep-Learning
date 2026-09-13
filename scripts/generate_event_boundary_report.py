#!/usr/bin/env python3
"""Regenerate results/event_level/event_metrics.csv and results/boundary_level/boundary_metrics.csv
(Tables 4 and 5) from a trained, adapted checkpoint's predictions on the LUDB test set.

Usage:
    python scripts/generate_event_boundary_report.py --config configs/A4.yaml \
        --checkpoint checkpoints/A4_seed1.pth --ludb-dir /path/to/ludb/1.0.1/data --seed 1
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_ludb import ludb_record  # noqa: E402
from src.preprocessing.windowing import build_partition, time_channel  # noqa: E402
from src.preprocessing.normalization import apply  # noqa: E402
from src.models.cnn_bilstm import RPeakGuidedML2, RPeakTimeML2  # noqa: E402
from src.training.train import predict  # noqa: E402
from src.analysis.event_metrics import event_level_metrics  # noqa: E402
from src.analysis.boundary_metrics import boundary_mae  # noqa: E402

MODELS = {'RPeakGuidedML2': RPeakGuidedML2, 'RPeakTimeML2': RPeakTimeML2}
SAMPLING_RATE_HZ = 250


def read_split(name):
    return [l.strip() for l in (ROOT / 'data' / 'splits' / name).read_text().splitlines() if l.strip()]


def resolve_norm_path(checkpoint_path, cfg):
    match = re.match(r'(.+)_seed(\d+)\.pth$', checkpoint_path.name)
    experiment, seed = match.group(1), match.group(2)
    if experiment == 'A4':
        experiment = cfg['training'].get('base_checkpoint', 'A3')
    return checkpoint_path.parent / f'{experiment}_seed{seed}_norm.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int, help='Used only to label the output rows.')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    post = cfg['window'].get('post_r')
    use_time_channel = cfg['window'].get('temporal_position_channel', False)

    norm_path = resolve_norm_path(args.checkpoint, cfg)
    norm = json.loads(norm_path.read_text())
    mean, std = norm['mean'], norm['std']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = MODELS[cfg['model']['architecture'].split(' ')[0]]
    model = model_cls().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    ludb_test = read_split('ludb_test.txt')
    X_test, y_test, _, _ = build_partition(ludb_test, ludb_record, post, args.ludb_dir)
    X_test_norm = apply(X_test, mean, std)
    X_test_norm = time_channel(X_test_norm, pre=120, post=post) if use_time_channel else X_test_norm[:, None]
    y_pred = predict(model, X_test_norm, device)

    event_rows, boundary_rows = [], []
    for class_index, wave in [(1, 'P'), (2, 'T')]:
        metrics = event_level_metrics(y_test, y_pred, class_index)
        event_rows.append({'seed': args.seed, 'wave': wave,
                            'sensitivity': round(metrics['sensitivity'], 4),
                            'ppv': round(metrics['ppv'], 4),
                            'f1': round(metrics['f1'], 4)})
        mae = boundary_mae(y_test, y_pred, class_index, SAMPLING_RATE_HZ)
        boundary_rows.append({'seed': args.seed, 'wave': wave, 'boundary': 'onset',
                               'mae_ms': round(mae['onset_mae_ms'], 2)})
        boundary_rows.append({'seed': args.seed, 'wave': wave, 'boundary': 'offset',
                               'mae_ms': round(mae['offset_mae_ms'], 2)})

    event_dir = ROOT / 'results' / 'event_level'
    boundary_dir = ROOT / 'results' / 'boundary_level'
    event_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    event_csv = event_dir / f'event_metrics_seed{args.seed}.csv'
    boundary_csv = boundary_dir / f'boundary_metrics_seed{args.seed}.csv'
    pd.DataFrame(event_rows).to_csv(event_csv, index=False)
    pd.DataFrame(boundary_rows).to_csv(boundary_csv, index=False)

    print(pd.DataFrame(event_rows).to_string(index=False))
    print(pd.DataFrame(boundary_rows).to_string(index=False))
    print(f'\nWrote {event_csv} and {boundary_csv}. '
          f'Merge across seeds into results/event_level/event_metrics.csv and '
          f'results/boundary_level/boundary_metrics.csv to compare against Tables 4/5.')


if __name__ == '__main__':
    main()
