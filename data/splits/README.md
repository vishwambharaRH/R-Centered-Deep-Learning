# Record-Level Splits

All splits below are **record-level**: partitioning was performed on whole QTDB/LUDB
records before any window construction, so no windows, beats, or leads from a given
record occur in more than one partition. This directly satisfies the leakage-safety
requirement raised in review — there is no possibility of the same record contributing
windows to both a training/adaptation partition and an evaluation partition.

| File | Dataset | Role | Records | Split seed |
| --- | --- | --- | ---: | ---: |
| `qtdb_train.txt` | QTDB | Model training (all experiments) | 84 | `random_state=7` |
| `qtdb_validation.txt` | QTDB | Model selection / validation | 21 | `random_state=7` |
| `ludb_adaptation.txt` | LUDB | Supervised adaptation (A4/R6 only) | 20 | `random_state=17` |
| `ludb_test.txt` | LUDB | Held-out cross-dataset evaluation | 180 | `random_state=17` |

## How the splits were produced

- **QTDB (84/21):** `sklearn.model_selection.train_test_split` on the 105 unique QTDB
  record IDs, `test_size=0.20`, `random_state=7`, `shuffle=True`. This split is reused,
  unchanged, across every experiment reported in the paper (R1–R6 and A0–A4) so that all
  configurations are trained and validated on identical data.
- **LUDB (20/180):** `train_test_split` on the 200 unique LUDB record IDs,
  `test_size=0.90` (i.e. `1 - R6_ADAPT_FRACTION` with `R6_ADAPT_FRACTION=0.10`),
  `random_state=17`, `shuffle=True`. The 20-record adaptation subset is only used for the
  supervised adaptation step in A4 (paper) / R6 (developmental). The remaining 180
  records are never used for training, adaptation, or model selection of any
  configuration — they are scored exactly once, after all hyperparameters and
  checkpoints were fixed.
- Exact reproduction: run `scripts/prepare_datasets.py`, which regenerates these four
  files deterministically from the seeds above and diffs the result against the files
  committed here.

## No leakage between adaptation and test

The `ludb_adaptation.txt` and `ludb_test.txt` lists are disjoint by construction (they
partition the same 200-record pool with no overlap) and neither list intersects
`qtdb_train.txt` or `qtdb_validation.txt`, since QTDB and LUDB record identifiers are
drawn from different databases entirely. No LUDB label or performance measurement from
the 180 held-out test records was used for model selection, early stopping, or
hyperparameter tuning at any stage.
