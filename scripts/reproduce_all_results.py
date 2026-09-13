#!/usr/bin/env python3
"""Reproduce as much of the paper's results as this script can fully automate.

What this script actually does, in order:
  1. Regenerate data/splits/*.txt from the raw QTDB/LUDB directories.
  2. Regenerate data/metadata/{window_counts,class_frequencies}.csv.
  3. Regenerate results/r_peak/r_peak_audit.csv (Table 6) — this needs no
     trained model at all, so it is fully automated here.
  4. Unless --skip-training: train A0, A1, A2, A3 for seeds 1/2/3, then A4
     (which loads each seed's A3 checkpoint and adapts on LUDB).
  5. Evaluate every trained checkpoint on the LUDB test set and write
     results/ablation/matched_ablation_A0_A4_regenerated.csv.

What this script does NOT do (each needs a human choice of checkpoint/seed,
so they are separate commands rather than folded in here):
  - Event-level / boundary-level regeneration (Tables 4/5) — run
    scripts/generate_event_boundary_report.py once per seed against a chosen
    A4 checkpoint, then combine the per-seed CSVs it writes.
  - R-error correlation regeneration (Table 7) — run
    scripts/generate_correlation_analysis.py against a chosen A4 checkpoint.
  - The R4 failure-analysis decoder (Section 4.3) — run
    scripts/run_r4_decoder.py against a trained R1 checkpoint.

Usage:
    python scripts/reproduce_all_results.py --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ['A0', 'A1', 'A2', 'A3', 'A4']
SEEDS = [1, 2, 3]


def run(cmd, capture=False):
    print('>>>', ' '.join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=True, capture_output=capture, text=True)
    if capture:
        print(result.stdout)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    parser.add_argument('--skip-training', action='store_true',
                         help='Only regenerate splits/metadata/R-peak audit; skip the (slow) A0-A4 training sweep.')
    args = parser.parse_args()

    run([sys.executable, ROOT / 'scripts' / 'prepare_datasets.py',
         '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir])
    run([sys.executable, ROOT / 'scripts' / 'generate_dataset_report.py',
         '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir])
    run([sys.executable, ROOT / 'scripts' / 'generate_r_peak_audit.py',
         '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir])

    if args.skip_training:
        print('\nSkipping training sweep (--skip-training). Splits, metadata, and the '
              'R-peak audit have been regenerated; ablation results have not.')
        return

    checkpoint_dir = ROOT / 'checkpoints'
    # A4 depends on A3's checkpoint existing for the same seed, so train A0-A3
    # for every seed before attempting any A4 run.
    for config in ['A0', 'A1', 'A2', 'A3']:
        for seed in SEEDS:
            run([sys.executable, ROOT / 'scripts' / 'train_ablation.py',
                 '--config', ROOT / 'configs' / f'{config}.yaml', '--seed', str(seed),
                 '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir,
                 '--checkpoint-dir', checkpoint_dir])
    for seed in SEEDS:
        run([sys.executable, ROOT / 'scripts' / 'train_ablation.py',
             '--config', ROOT / 'configs' / 'A4.yaml', '--seed', str(seed),
             '--qtdb-dir', args.qtdb_dir, '--ludb-dir', args.ludb_dir,
             '--checkpoint-dir', checkpoint_dir])

    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            checkpoint = checkpoint_dir / f'{config}_seed{seed}.pth'
            result = run([sys.executable, ROOT / 'scripts' / 'evaluate_ablation.py',
                          '--config', ROOT / 'configs' / f'{config}.yaml',
                          '--checkpoint', checkpoint, '--ludb-dir', args.ludb_dir], capture=True)
            macro_f1 = float(result.stdout.split('Macro F1:')[1].split()[0])
            rows.append({'configuration': config, 'seed': seed, 'ludb_macro_f1': macro_f1})

    df = pd.DataFrame(rows)
    summary = df.groupby('configuration')['ludb_macro_f1'].agg(['mean', 'std']).reindex(CONFIGS)
    out_path = ROOT / 'results' / 'ablation' / 'matched_ablation_A0_A4_regenerated.csv'
    df.to_csv(out_path.with_name('matched_ablation_A0_A4_regenerated_per_seed.csv'), index=False)
    summary.to_csv(out_path)
    print(summary)
    print(f'\nWrote {out_path} — compare against the committed reference, '
          f'results/ablation/matched_ablation_A0_A4.csv (Table 2).')
    print('\nFor Tables 4/5/7 and the R4 decoder, run generate_event_boundary_report.py, '
          'generate_correlation_analysis.py, and run_r4_decoder.py against a chosen checkpoint.')


if __name__ == '__main__':
    main()
