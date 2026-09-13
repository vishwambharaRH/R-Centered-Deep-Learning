"""R-aware hard decoder for the R4 failure-analysis experiment.

R4 (Section 4.3, Table 3 of the paper) applies a frozen, rule-based decoder to
a frozen R1 model's raw per-sample class probabilities. The rule: predict the
T class only where the model's T-class probability clears a fixed threshold
(locked at 0.40 on the QTDB validation set — chosen once, not re-tuned per
record); everywhere else, fall back to the model's ordinary argmax over
{Background, P}. This makes T-wave prediction strictly harder to trigger than
plain argmax without touching the P/Background decision at all, which is
consistent with the paper's report of the frozen-decoder configuration
substantially *degrading* Macro F1 (0.9026 -> 0.7328 on QTDB) by suppressing
valid T predictions rather than fixing errors.

This module is a documented reconstruction of that rule from the paper's
description (Section 4.3) — the original notebook cell that selected/applied
the 0.40 threshold was not recovered verbatim. Function names and the
threshold-selection procedure below are written so the rule is exactly
reproducible from this file alone; treat the reconstruction as the
specification of R4's decoding rule going forward, not a byte-for-byte replay
of an unrecovered original implementation.
"""
import numpy as np
import torch

BACKGROUND, P, T = 0, 1, 2


def softmax_probabilities(model, features, device, batch_size=64):
    """Per-sample class probabilities (N, T, 3) from a frozen model."""
    model.eval()
    probs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(batch), dim=2).cpu().numpy())
    return np.concatenate(probs)


def select_t_threshold(probabilities, labels, thresholds=np.linspace(0.05, 0.95, 19)):
    """Locate the T-probability threshold maximizing T-class F1 on a validation set.

    This is how R4's threshold is meant to be *locked*: run once on the QTDB
    validation set, then frozen and reused unchanged at test time (never
    re-tuned per record or per dataset).
    """
    from sklearn.metrics import f1_score

    flat_probs_t = probabilities[..., T].ravel()
    flat_labels = labels.ravel()
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        pred_is_t = flat_probs_t >= threshold
        f1 = f1_score(flat_labels == T, pred_is_t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return float(best_threshold), float(best_f1)


def decode(probabilities, t_threshold):
    """Apply the locked R-aware hard decoder to a batch of class probabilities.

    Rule: predict T where ``P(T) >= t_threshold``; elsewhere, argmax over
    {Background, P} only (T is excluded from that argmax, since it has
    already been decided by the threshold test).
    """
    is_t = probabilities[..., T] >= t_threshold
    bg_p_pred = np.argmax(probabilities[..., [BACKGROUND, P]], axis=-1)  # 0 or 1, already correct label ids
    return np.where(is_t, T, bg_p_pred)
