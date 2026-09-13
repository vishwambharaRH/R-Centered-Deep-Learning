# Figures

Extracted directly from `paper/ICBBE-17.pdf`:

- `architecture.png` — Fig. 1, the CNN-BiLSTM baseline architecture, **as published**.
- `r_guided_framework.png` — Fig. 2, the R-guided delineation framework.
- `qtdb_example.png` — Fig. 3, representative QTDB delineation result (ground truth vs. R3 baseline prediction).
- `ludb_example.png` — Fig. 4, representative LUDB delineation result (ground truth vs. R3 baseline vs. R6/proposed framework prediction).

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
