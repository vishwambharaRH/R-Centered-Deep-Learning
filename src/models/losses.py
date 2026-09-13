"""Focal loss variants used in the R1-R3 baseline-development comparison.

R1 used plain cross-entropy. R2 used class-weighted focal loss (``alpha`` set to
inverse class frequency) and increased minority-class recall at the cost of
precision (Macro F1 dropped to 0.8420 on QTDB / 0.7604 on LUDB). R3 used
unweighted focal loss, which restored the precision/recall balance and gave the
strongest baseline (Macro F1 0.9061 QTDB / 0.8236 LUDB) — R3 was therefore
selected as the reference baseline for the physiological-guidance experiments
(R4-R6) and its unweighted cross-entropy-style training regime carries forward
into the A0-A4 matched ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with optional per-class weighting (``alpha``)."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', alpha if alpha is not None else None, persistent=False)

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class UnweightedFocalLoss(FocalLoss):
    def __init__(self, gamma=2.0):
        super().__init__(gamma=gamma, alpha=None)


def inverse_frequency_weights(class_counts, device=None):
    """``len(targets) / (num_classes * count_c)`` per class, as used for R2's alpha."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return weights.to(device) if device is not None else weights
