#!/usr/bin/env python3
"""Generate data/metadata/{window_counts,class_frequencies}.csv and a summary
report/JSON from the exact preprocessing code in src/preprocessing and
src/r_peak, run against the record-level splits in data/splits/.

Usage:
    python scripts/generate_dataset_report.py --qtdb-dir /path/to/qtdb/1.0.0 \
        --ludb-dir /path/to/ludb/1.0.1/data
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_qtdb import qtdb_record  # noqa: E402
from src.preprocessing.load_ludb import ludb_record  # noqa: E402
from src.preprocessing.windowing import build_partition, POST, POST_EXTENDED  # noqa: E402

CLASS_NAMES = ['Background', 'P', 'T']


def read_split(name):
    return [line.strip() for line in (ROOT / 'data' / 'splits' / name).read_text().splitlines() if line.strip()]


def per_record_windows(record_names, builder, post, data_dir):
    rows = []
    for record_name in record_names:
        try:
            x, y = builder(record_name, post, data_dir)
        except Exception as exc:  # pragma: no cover
            print(f'  ! skipped {record_name}: {exc}', file=sys.stderr)
            continue
        rows.append({'record': record_name, 'windows': len(x), 'window_length': post + 120})
    return rows


def class_counts(labels):
    counts = np.bincount(labels.ravel(), minlength=3)[:3]
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    args = parser.parse_args()

    partitions = {
        'QTDB': {
            'train': (read_split('qtdb_train.txt'), qtdb_record, args.qtdb_dir),
            'validation': (read_split('qtdb_validation.txt'), qtdb_record, args.qtdb_dir),
        },
        'LUDB': {
            'adaptation': (read_split('ludb_adaptation.txt'), ludb_record, args.ludb_dir),
            'test': (read_split('ludb_test.txt'), ludb_record, args.ludb_dir),
        },
    }

    per_record_rows = []
    summary_rows = []
    class_rows = []

    for post, tag in [(POST, 240), (POST_EXTENDED, 320)]:
        for dataset, parts in partitions.items():
            for partition, (records, builder, data_dir) in parts.items():
                print(f'[{tag}] {dataset} {partition}: {len(records)} records')
                X, Y, ids, skipped = build_partition(records, builder, post, data_dir)
                counts = class_counts(Y)
                total = int(counts.sum())

                for rec in records:
                    n_windows = int((ids == rec).sum())
                    per_record_rows.append({
                        'dataset': dataset, 'partition': partition, 'record': rec,
                        'windows': n_windows, 'window_length': post + 120,
                    })
                summary_rows.append({
                    'dataset': dataset, 'partition': partition,
                    'total_records': len(records), 'total_windows': len(X),
                    'window_length': post + 120,
                })
                for cls_idx, cls_name in enumerate(CLASS_NAMES):
                    class_rows.append({
                        'dataset': dataset, 'partition': partition, 'window_length': post + 120,
                        'class': cls_name, 'samples': int(counts[cls_idx]),
                        'percentage': round(100.0 * counts[cls_idx] / total, 4) if total else 0.0,
                    })
                if skipped:
                    print(f'    skipped: {skipped}', file=sys.stderr)

    meta_dir = ROOT / 'data' / 'metadata'
    meta_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(per_record_rows).to_csv(meta_dir / 'window_counts_per_record.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(meta_dir / 'window_counts.csv', index=False)
    pd.DataFrame(class_rows).to_csv(meta_dir / 'class_frequencies.csv', index=False)

    (meta_dir / 'preprocessing_summary.json').write_text(json.dumps({
        'window_lengths': {'primary': 360, 'extended': 440},
        'pre_r_samples': 120,
        'post_r_samples': {'primary': 240, 'extended': 320},
        'qtdb_sampling_rate_hz': 250,
        'ludb_native_sampling_rate_hz': 500,
        'ludb_processing_sampling_rate_hz': 250,
        'qtdb_annotation_source': 'pu0',
        'ludb_annotation_source': 'ii',
        'r_peak_detector': 'pan_tompkins',
        'normalization': 'mean/std computed on QTDB training windows only',
        'summary': summary_rows,
    }, indent=2))

    print('\nWrote data/metadata/window_counts.csv, window_counts_per_record.csv, '
          'class_frequencies.csv, preprocessing_summary.json')


if __name__ == '__main__':
    main()
