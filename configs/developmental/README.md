# Developmental R1-R6 Configs

These document the sequential framework evolution that led to the matched
A0-A4 ablation (see `configs/README.md` and the paper's Section 3.6.1). They
are historical records of runs already performed with the original training
script (`src/training/train.py`'s predecessor), not specs meant to be re-run
through the current `scripts/train_ablation.py` pipeline.

**Known discrepancy — checkpoint selection.** Every `checkpoint_selection:
best_validation_loss` field here reflects what those historical runs actually
did (selecting the epoch with the lowest QTDB validation loss), consistent
with the `best_epoch`/`best_validation_loss` values reported in each file.
The paper's Section 3.5 states model selection "was based on the highest
Macro F1 achieved on the QTDB validation set" — that criterion is what
`src/training/train.py` implements for the current A0-A4 pipeline
(`configs/A0.yaml` … `configs/A3.yaml` all use
`checkpoint_selection: best_validation_macro_f1`), but it does not describe
how these earlier R1-R6 runs were actually selected. This file exists so
that discrepancy is documented rather than silently inconsistent.
