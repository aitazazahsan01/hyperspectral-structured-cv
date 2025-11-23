"""
train.py

Shared training/evaluation loop for both models in this repo. Deliberately
generic over the model -- it doesn't know or care whether it's training the
pixel baseline or the patch-based structured model, only that the model
takes a batch of inputs and returns logits. That keeps the ablation script
honest: both models are trained and scored through the exact same code
path, so any accuracy difference reflects the models/data, not a
difference in training procedure.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model: nn.Module, train_loader: DataLoader, epochs: int, lr: float = 1e-3, device: str = "cpu") -> nn.Module:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Returns overall accuracy on `loader`."""
    model.to(device)
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total
