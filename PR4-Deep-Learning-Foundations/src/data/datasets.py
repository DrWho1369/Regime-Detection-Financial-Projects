from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class StackedWindowDataset(Dataset):
    """
    Dataset for stacked window tensors produced by tensor_builder.

    Expects X of shape (samples, lookback, n_features) and y of shape (samples,).
    Returns tensors shaped for a 2D CNN: (1, lookback, n_features).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (samples, lookback, n_features), got {X.shape}.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with the same length as X samples.")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].unsqueeze(0)  # (1, lookback, n_features)
        y = self.y[idx]
        return x, y


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to wrap numpy arrays into PyTorch DataLoaders.
    """
    train_ds = StackedWindowDataset(X_train, y_train)
    val_ds = StackedWindowDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader

