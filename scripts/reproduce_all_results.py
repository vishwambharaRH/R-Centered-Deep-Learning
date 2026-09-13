#!/usr/bin/env python3
"""Orchestrate a full reproduction: splits -> dataset report -> A0-A4 training -> evaluation -> results CSVs.

This is the top-level entry point referenced in the README. It shells out to
the other scripts/ so each stage can also be run independently.

Usage:
    python scripts/reproduce_all_results.py --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ['A0', 'A1', 'A2', 'A3', 'A4']
SEEDS = [1, 2, 3]


def run(cmd):
    print('>>>', ' '.join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--skip-training', action='store_true',
                         help='Only regenerate splits/metadata; skip the (slow) A0-A4 training sweep.')
    args = parser.parse_args()

    run([sys.executable, ROOT / 'scripts' / 'prepare_datasets.py',
         '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir])
    run([sys.executable, ROOT / 'scripts' / 'generate_dataset_report.py',
         '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir])

    if args.skip_training:
        print('Skipping training sweep (--skip-training). Splits and metadata regenerated.')
        return

    for config in CONFIGS:
        for seed in SEEDS:
            run([sys.executable, ROOT / 'scripts' / 'train_ablation.py',
                 '--config', ROOT / 'configs' / f'{config}.yaml', '--seed', str(seed),
                 '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir])

    print('\nTraining sweep complete. Aggregate checkpoints/*.pth evaluation results '
          'into results/ablation/matched_ablation_A0_A4.csv to compare against the '
          'committed reference values.')


if __name__ == '__main__':
    main()
