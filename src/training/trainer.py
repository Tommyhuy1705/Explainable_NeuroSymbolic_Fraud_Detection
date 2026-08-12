"""Deterministic PyTorch training loop for imbalanced binary classification."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_pr_auc: float
    device: str


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    features = np.array(X, dtype=np.float32, copy=True)
    labels = np.array(y, dtype=np.float32, copy=True)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_torch_proba(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 8192,
    device: str | None = None,
) -> np.ndarray:
    target_device = resolve_device(device)
    model = model.to(target_device)
    model.eval()
    probabilities: list[np.ndarray] = []
    dummy = np.zeros(len(X), dtype=np.float32)
    for features, _ in _loader(X, dummy, batch_size, shuffle=False):
        logits = model(features.to(target_device, non_blocking=True))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(float)


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: dict[str, Any],
    seed: int = 42,
    device: str | None = None,
) -> TrainResult:
    """Train with class weighting, OneCycleLR, and validation PR-AUC early stopping."""
    set_global_seed(seed)
    target_device = resolve_device(device)
    model = model.to(target_device)
    batch_size = int(config.get("batch_size", 2048))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 4))
    train_loader = _loader(X_train, y_train, batch_size, shuffle=True)

    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=target_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config.get("learning_rate", 1e-3)),
        epochs=epochs,
        steps_per_epoch=max(len(train_loader), 1),
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, labels in train_loader:
            features = features.to(target_device, non_blocking=True)
            labels = labels.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item()) * len(labels)
            total_rows += len(labels)

        validation_probabilities = predict_torch_proba(model, X_validation, batch_size, str(target_device))
        validation_score = float(average_precision_score(y_validation, validation_probabilities))
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total_rows, 1),
            "validation_pr_auc": validation_score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        if validation_score > best_score + 1e-6:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_pr_auc=float(best_score),
        device=str(target_device),
    )
