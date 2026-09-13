#!/usr/bin/env python3
"""Regenerate data/splits/*.txt deterministically from raw QTDB/LUDB directories.

QTDB is split 84/21 (train/validation) with random_state=7. LUDB is split
20/180 (adaptation/test) with random_state=17. Both splits are performed at
the record level, matching Section 3.2.2 of the paper. Running this script
against a fresh QTDB/LUDB download should reproduce the exact files already
committed under data/splits/.
"""
import argparse
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.load_qtdb import list_qtdb_records  # noqa: E402
from src.preprocessing.load_ludb import list_ludb_records  # noqa: E402

QTDB_SPLIT_SEED = 7
LUDB_SPLIT_SEED = 17
LUDB_ADAPT_FRACTION = 0.10


def write_list(path, records, numeric=False):
    key = (lambda r: int(r)) if numeric else (lambda r: r)
    path.write_text('\n'.join(sorted(records, key=key)) + '\n')
    print(f'wrote {path} ({len(records)} records)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qtdb-dir', required=True, type=Path)
    parser.add_argument('--ludb-dir', required=True, type=Path)
    args = parser.parse_args()

    splits_dir = ROOT / 'data' / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    qtdb_records = list_qtdb_records(args.qtdb_dir)
    qtdb_train, qtdb_val = train_test_split(
        qtdb_records, test_size=0.20, random_state=QTDB_SPLIT_SEED, shuffle=True)
    write_list(splits_dir / 'qtdb_train.txt', qtdb_train)
    write_list(splits_dir / 'qtdb_validation.txt', qtdb_val)

    ludb_records = list_ludb_records(args.ludb_dir)
    ludb_adapt, ludb_test = train_test_split(
        ludb_records, test_size=1 - LUDB_ADAPT_FRACTION, random_state=LUDB_SPLIT_SEED, shuffle=True)
    write_list(splits_dir / 'ludb_adaptation.txt', ludb_adapt, numeric=True)
    write_list(splits_dir / 'ludb_test.txt', ludb_test, numeric=True)

    print('\nDone. Diff against the committed data/splits/*.txt to confirm an exact match.')


if __name__ == '__main__':
    main()
