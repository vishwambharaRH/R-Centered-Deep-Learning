"""Training loop shared by every configuration (A0-A3, R1-R5).

Adam optimizer, learning rate 1e-3, best-validation-loss checkpointing (no
early stopping — every configuration always trains for its full epoch budget:
15 epochs for the matched A0-A3 ablation, 30 epochs for the R1-R5 developmental
runs). Data loaders are seeded per-run so the three-seed sweep (seeds 1, 2, 3)
covers both weight initialization and minibatch ordering.
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 64


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader(features, labels, seed, shuffle, batch_size=BATCH_SIZE):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def run_training_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features).permute(0, 2, 1)  # (N, C, T) for CE/focal loss
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            batches += 1
    return total_loss / max(1, batches)


def train_model(model, X_train, y_train, X_val, y_val, criterion, seed, num_epochs, device,
                 lr=1e-3, checkpoint_path=None, log_fn=print):
    """Train ``model`` for ``num_epochs``, keeping the best-validation-loss state dict."""
    set_global_seed(seed)
    train_loader = make_loader(X_train, y_train, seed, shuffle=True)
    val_loader = make_loader(X_val, y_val, seed, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_state = None
    best_epoch = None
    for epoch in range(1, num_epochs + 1):
        train_loss = run_training_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_training_epoch(model, val_loader, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log_fn:
            log_fn(f'seed={seed} epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}')
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return model, {'best_epoch': best_epoch, 'best_validation_loss': best_loss}


def predict(model, features, device, batch_size=BATCH_SIZE):
    model.eval()
    preds = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(batch).argmax(2).cpu().numpy())
    return np.concatenate(preds)
