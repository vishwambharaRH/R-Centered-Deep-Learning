"""Supervised LUDB adaptation for A4 (paper) / R6 (developmental).

The corresponding best A3 (or R5, for R6) checkpoint for each seed is used as
initialization, followed by 8 epochs of supervised fine-tuning on the 20-record
LUDB adaptation set with a 10x lower learning rate (1e-4) than pretraining.
This epoch budget was fixed in advance and not tuned against the 180-record
held-out test set.
"""
import torch

from .train import make_loader, run_training_epoch

ADAPT_EPOCHS = 8
ADAPT_LR = 1e-4


def adapt_model(model, X_adapt, y_adapt, criterion, seed, device, checkpoint_path=None, log_fn=print):
    loader = make_loader(X_adapt, y_adapt, seed, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAPT_LR)
    for epoch in range(1, ADAPT_EPOCHS + 1):
        train_loss = run_training_epoch(model, loader, criterion, device, optimizer)
        if log_fn:
            log_fn(f'[adapt] seed={seed} epoch={epoch:02d} train_loss={train_loss:.4f}')
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return model
