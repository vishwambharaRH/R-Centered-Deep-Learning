# Physiologically Guided R-Centered Deep Learning for ECG Wave Delineation Across Heterogeneous Databases

Official implementation and reproducibility materials for:

> Shruti Kanakeri, Vishwambhara R Hebbalalu, Vrishank N Amembal, Vyshnavi Karanam,
> Yukthi N Sharma, and Ashok Kumar Patil. *Physiologically Guided R-Centered Deep
> Learning for ECG Wave Delineation Across Heterogeneous Databases.* ICBBE 2026.

## Overview

This repository investigates whether incorporating physiological structure into
deep learning improves ECG waveform delineation, particularly for the P and T
waves, under cross-dataset conditions. The central idea is to use the **R
peak as a physiologically meaningful temporal reference** and combine this
representation with a CNN–BiLSTM sequence-labeling architecture. Experiments
were conducted on the QT Database (QTDB) and the Lobachevsky University
Electrocardiography Database (LUDB), using strictly record-level partitions.

## Key Findings

- R-centered representation improved cross-dataset LUDB Macro F1: 0.7999 → 0.8202 (A0 → A1)
- Extended post-R temporal context provided a further gain: 0.8202 → 0.8308 (A1 → A2)
- The normalized temporal-position channel did **not** help: 0.8308 → 0.8263 (A2 → A3)
- Supervised adaptation on 20 LUDB records improved performance on the 180 held-out
  records: 0.8263 → 0.8415 ± 0.0073 (A3 → A4)
- Physiological guidance is beneficial, but more physiological constraint is not
  inherently better — see the R4 failure-analysis experiment (`configs/developmental/R4.yaml`).

## Repository Structure

```
ecg-r-centered-delineation/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/                 # A0-A4 matched ablation + R1-R6 developmental configs
│   └── developmental/
├── data/
│   ├── README.md            # dataset provenance, leakage-prevention notes
│   ├── splits/               # exact record-level train/val/adaptation/test lists
│   └── metadata/             # generated window counts, class frequencies, summary
├── src/
│   ├── preprocessing/        # QTDB/LUDB loading, annotation parsing, windowing
│   ├── r_peak/                # Pan-Tompkins detector used as physiological anchor
│   ├── models/                # CNN-BiLSTM architecture + focal loss variants
│   ├── training/              # training loop, LUDB adaptation, evaluation
│   └── analysis/              # event-level, boundary-level, correlation analysis
├── scripts/                  # dataset prep, report generation, ablation runners
├── results/                  # machine-readable CSVs for every paper table
├── figures/
└── paper/
    └── ICBBE-17.pdf
```

## Reproducibility

### 1. Obtain the datasets

QTDB and LUDB are not redistributed in this repository (see `data/README.md`).
Download them from PhysioNet:

- QT Database: https://physionet.org/content/qtdb/1.0.0/
- LUDB: https://physionet.org/content/ludb/1.0.1/

### 2. Install dependencies

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

### 3. Regenerate splits and the dataset report

```bash
python scripts/prepare_datasets.py --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data
python scripts/generate_dataset_report.py --qtdb-dir /path/to/qtdb/1.0.0 --ludb-dir /path/to/ludb/1.0.1/data
```

This deterministically reproduces `data/splits/*.txt` (QTDB `random_state=7`,
LUDB `random_state=17`) and `data/metadata/{window_counts,class_frequencies}.csv`
from the raw data, using the exact preprocessing and R-peak detection code in
`src/`. The committed metadata files in this repository were generated this
way and match the counts printed by the original training runs (e.g. QTDB
train: 87,111 windows from 84 records; LUDB held-out test: 1,710 windows from
180 records at the 360-sample window length).

### 4. Train a configuration

```bash
python scripts/train_ablation.py --config configs/A1.yaml --seed 1
```

### 5. Evaluate / reproduce a results table

```bash
python scripts/evaluate_ablation.py --config configs/A1.yaml --checkpoint <path>
python scripts/reproduce_all_results.py   # regenerates every CSV under results/
```

## Dataset Splits

| Dataset | Partition | Records | Windows (360-sample) | Windows (440-sample) |
| --- | --- | ---: | ---: | ---: |
| QTDB | Train | 84 | 87,111 | 87,088 |
| QTDB | Validation | 21 | 22,799 | 22,793 |
| LUDB | Adaptation | 20 | 194 | 183 |
| LUDB | Held-out test | 180 | 1,710 | 1,655 |

See `data/splits/` for the exact record IDs and `data/metadata/` for the full,
generated per-record breakdown and class frequencies.

## Experimental Configurations

**Matched A0-A4 ablation** (`configs/`, common protocol: 15 epochs, seeds 1/2/3,
QTDB train/validation, evaluation on the 180-record LUDB test set):

| Config | Description | LUDB Macro F1 |
| --- | --- | --- |
| A0 | Fixed, non-R-centered baseline | 0.7999 ± 0.0042 |
| A1 | R-centered representation | 0.8202 ± 0.0009 |
| A2 | A1 + extended post-R context (POST320) | 0.8308 ± 0.0078 |
| A3 | A2 + normalized temporal-position channel | 0.8263 ± 0.0061 |
| A4 | A3 + supervised LUDB adaptation (20 records) | 0.8415 ± 0.0073 |

**Developmental R1-R6** (`configs/developmental/`, sequential framework evolution,
presented as evidence rather than a controlled ablation — see Section 3.6.1 of the paper).

## Results

Machine-readable results for every table in the paper are under `results/`
(see `results/README.md` for a full index): the matched ablation, R1-R6
developmental summary, event-level P/T metrics, boundary-error MAE, the
independent R-peak audit, and the R-error/downstream-F1 correlation analysis.

## Reproducibility Checklist

| Requirement | Location |
| --- | --- |
| Exact QTDB records | `data/splits/qtdb_*.txt` |
| Exact LUDB records | `data/splits/ludb_*.txt` |
| Record-level splits, leakage prevention | `data/splits/README.md` |
| Window counts | `data/metadata/window_counts.csv`, `window_counts_per_record.csv` |
| Class frequencies | `data/metadata/class_frequencies.csv` |
| Preprocessing implementation | `src/preprocessing/` |
| R-peak detector | `src/r_peak/pan_tompkins.py` |
| CNN-BiLSTM architecture | `src/models/cnn_bilstm.py` |
| Loss functions | `src/models/losses.py` |
| Training parameters, seeds | `configs/`, `src/training/train.py` |
| LUDB adaptation procedure | `src/training/adapt.py`, `configs/A4.yaml` |
| Event-level / boundary-level / correlation analysis | `src/analysis/` |
| Reported results | `results/` |

## Datasets

- QT Database (QTDB)
- Lobachevsky University Database (LUDB)

Both datasets are publicly available through PhysioNet and should be obtained
according to their original licensing and usage requirements; this repository
does not redistribute them (`data/README.md`).

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
