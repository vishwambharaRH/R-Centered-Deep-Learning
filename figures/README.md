# Figures

Extracted directly from `paper/ICBBE-17.pdf`:

- `architecture.png` — Fig. 1, the CNN-BiLSTM baseline architecture, **as published**.
- `r_guided_framework.png` — Fig. 2, the R-guided delineation framework.
- `qtdb_example.png` — Fig. 3, **as published**: representative QTDB delineation result (ground truth vs. R3 baseline prediction), unlabeled colors, captioned "a QTDB validation beat."
- `ludb_example.png` — Fig. 4, **as published**: representative LUDB delineation result (ground truth vs. R3 baseline vs. R6/proposed framework prediction), unlabeled colors.

## Corrected Figures 3 & 4 (`misc/changesToFig3and4.txt`)

`qtdb_example_corrected.png` and `ludb_example_corrected.png` regenerate the
exact same representative examples as Figs. 3 and 4 (same QTDB/LUDB indices,
same R3/R6 checkpoints — `scripts/generate_figures_3_4.py`), with three fixes:

1. **Explicit legend** — Background / P wave / T wave. As published, the
   color bars carried no legend at all, and the underlying colormap even had
   an unused fourth "QRS" slot; QRS is never a model class (it's folded into
   Background per Section 3.1), so that slot is removed entirely here.
2. **Panel titles name their classes** — "Ground Truth (Background / P / T)",
   "R3 Prediction (Background / P / T)", etc., instead of an unexplained
   color bar under a bare "Ground Truth" / "R3 Prediction" label.
3. **Caption wording** — both segments span more than one cardiac cycle, so
   neither is called "a beat" anymore; both are now "representative ... ECG
   segment."

**Also corrected here: which color is P and which is T.** Reading the
original plotting code (`draw_mask`'s colormap) against the label encoding
(`to_pt_labels`: label 1 = P, label 2 = T) shows label 1 → `#66c2a5` (teal)
and label 2 → `#fc8d62` (orange) — i.e. **P wave is teal, T wave is orange**.
This is also confirmed physiologically in the images themselves (the teal
segment always precedes the sharp QRS spike, the orange segment always
follows it, matching P-QRS-T order). The rest of this repository's figures
(the class-frequency chart in `full_analytics.html`) now use this same
mapping — do not swap P/T colors when adding new charts.

## Known discrepancy: Fig. 1 vs. the actual implementation

`architecture.png` (Fig. 1, as published) labels the input `N = 512` samples
and visually depicts each convolutional block as `Conv1D → ReLU → BatchNorm`.
Neither matches the code that actually produced every result in this
repository (`src/models/cnn_bilstm.py`, and the original
`colab_l4_ecg_train.py`):

- Actual window lengths are **360** (PRE=120/POST=240) or **440**
  (PRE=120/POST=320) samples — never 512.
- The actual per-block order is **`Conv1D → BatchNorm → ReLU`**.

`architecture_corrected.svg` is a plain schematic matching the real code, for
reference alongside the published figure — it does not replace Fig. 1 as
published; both are kept, with this discrepancy disclosed.
