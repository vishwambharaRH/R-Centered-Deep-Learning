# Results

All files here are machine-readable and correspond directly to tables in the
paper. Unless noted otherwise, all evaluation is on the 180-record LUDB
held-out test set (`data/splits/ludb_test.txt`).

| File | Paper reference |
| --- | --- |
| `ablation/matched_ablation_A0_A4.csv` | Table 2 — matched A0-A4 ablation |
| `developmental/R1_R6_summary.csv` | Table 3 — R1-R6 developmental configurations |
| `event_level/event_metrics.csv` | Table 4 — event-level P/T delineation (final A4/R6 configuration) |
| `boundary_level/boundary_metrics.csv` | Table 5 — P/T boundary MAE in ms (matched events only) |
| `r_peak/r_peak_audit.csv` | Table 6 — independent R-peak detector audit (regenerated live by `scripts/generate_r_peak_audit.py` against the real QTDB/LUDB data — see note below) |
| `correlations/r_error_correlations.csv` | Table 7 — Spearman correlations, R-localization error vs. downstream F1 |
| `qualitative_difficult_records.csv` | Section 4.8 — records with severe R-detection error and P-wave degradation |

## Notes on evaluation protocol differences

- **Event-level matching:** a predicted P/T event counts as a match to a
  reference event within 25 samples (100 ms at 250 Hz); unmatched predictions
  and references count as false positives / false negatives respectively.
- **Boundary MAE** is computed only over matched events (Section 4.5) and
  therefore complements, rather than replaces, event-level sensitivity/PPV.
- **R1/R5 vs. R6/A4:** R1 and R5 evaluate directly on all 200 LUDB records
  (no adaptation), while R6/A4 adapt on 20 records and evaluate on the
  remaining 180. These two protocols are not directly comparable — see
  `configs/developmental/R5.yaml` and Section 4.3 of the paper.
- **Significance codes** in `r_error_correlations.csv`: `*` p<0.05, `**`
  p<0.01, `***` p<0.001 (Spearman correlation, n=180 LUDB test records).

## Regenerating results from raw data

Two files in this tree have been verified by actually running their
generator scripts against the real, local QTDB/LUDB data (not just
transcribed from the paper):

- **`r_peak/r_peak_audit.csv`** — `scripts/generate_r_peak_audit.py`,
  auditing all 105 QTDB and all 200 LUDB records, reproduces the paper's
  sensitivity, PPV, and mean timing error to 4 decimal places (QTDB
  0.9780/0.9863/13.31 ms; LUDB 0.9753/0.7971/3.07 ms — exact matches). F1 is
  reproduced to within ~0.002 (QTDB 0.9824 vs. paper 0.9804; LUDB 0.8815 vs.
  paper 0.8800): the exact F1 aggregation convention across records (this
  script pools TP/FP/FN before computing F1, i.e. a micro-average) was not
  fully recoverable from the archived notebooks, so a small, disclosed
  discrepancy remains in that one derived column. Per-record detail is in
  `r_peak/r_peak_audit_{qtdb,ludb}_per_record.csv`.
- **`ablation/matched_ablation_A0_A4.csv`**, **`event_level/*`**,
  **`boundary_level/*`**, **`correlations/*`** are transcribed directly from
  the paper's Tables 2, 4, 5, and 7 (seed-level and per-record raw numbers,
  not just the aggregate). `scripts/generate_correlation_analysis.py` and a
  trained checkpoint can regenerate the correlation table from scratch;
  event-level/boundary-level regeneration follows the same pattern using
  `src/analysis/event_metrics.py` and `src/analysis/boundary_metrics.py`
  against a checkpoint's LUDB test predictions.
