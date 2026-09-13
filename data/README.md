# Dataset and Splits

This repository does not redistribute QTDB or LUDB. Both datasets must be obtained from
their respective PhysioNet sources (see links below) and placed locally; the
preprocessing scripts in `src/preprocessing/` then read them directly.

## QT Database (QTDB)

- Source: [PhysioNet QT Database](https://physionet.org/content/qtdb/1.0.0/)
- Total records: 105
- Training: 84 records
- Validation: 21 records
- Sampling rate: 250 Hz (native)
- Lead: first recorded channel (MLII)
- Annotation source: `pu0`

## Lobachevsky University Database (LUDB)

- Source: [PhysioNet LUDB](https://physionet.org/content/ludb/1.0.1/)
- Total records: 200
- Original sampling rate: 500 Hz
- Processing sampling rate: 250 Hz (resampled via `scipy.signal.resample_poly`,
  `up=1, down=2`; annotation sample indices are scaled by the same factor)
- Lead: II (matched to QTDB's MLII)
- Annotation source: `ii`

## Adaptation / Test Split

- Adaptation: 20 records (10%)
- Held-out test: 180 records
- Split performed at record level before window construction, `random_state=17`.
- No held-out LUDB labels or performance were used for model selection at any stage.

## Label formulation

Raw annotations are 4-class (`Background`, `P`, `QRS`, `T`). For the delineation task
studied in this repository, QRS-labeled samples are mapped to `Background`, giving the
three-class formulation used throughout: `{Background, P, T}`. The reported sample-wise
Macro F1 is the average F1 across these three classes.

## Preprocessing summary

- Normalization statistics (mean/std) are computed **only** on the QTDB training
  partition and then applied unchanged to QTDB validation and all LUDB data.
- R-centered windows are built around R peaks detected with a Pan–Tompkins detector
  (`src/r_peak/pan_tompkins.py`); the primary configuration uses 120 pre-R / 240 post-R
  samples (360-sample windows), with an extended-context variant using 120 pre-R / 320
  post-R samples (440-sample windows).
- See `data/splits/README.md` for the exact record lists and `data/metadata/` for
  generated window counts and class frequencies.

## Reproducing this data layer

```bash
python scripts/prepare_datasets.py --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data
python scripts/generate_dataset_report.py
```

This regenerates `data/splits/*.txt`, `data/metadata/window_counts.csv`,
`data/metadata/class_frequencies.csv`, and `data/metadata/preprocessing_summary.json`
from the raw PhysioNet files using the fixed seeds documented in
`data/splits/README.md`.
