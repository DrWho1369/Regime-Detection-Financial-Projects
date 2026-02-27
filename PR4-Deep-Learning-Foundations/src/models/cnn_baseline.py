from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class DirectionalBigErrorLoss(nn.Module):
    """
    Directional Big Error (DBE) loss.

    Heavily penalises forecasts that get the direction (sign) of the return
    wrong, while treating correct-direction magnitude errors more gently.

    L_DBE(y, y_hat) = mean( (y - y_hat)^2 * (1 + alpha * 1[sign mismatch]) )
    """

    def __init__(self, alpha: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            y_true = y_true.view_as(y_pred)

        mse = (y_pred - y_true) ** 2
        wrong_direction = (torch.sign(y_pred) != torch.sign(y_true)).float()
        penalty = 1.0 + self.alpha * wrong_direction
        loss = torch.mean(mse * penalty)
        return loss


@dataclass
class CnnConfig:
    n_features: int
    lookback: int = 10
    conv_channels: int = 16
    hidden_dim: int = 64
    dropout: float = 0.2


class CnnBaseline(nn.Module):
    """
    Simple 2D CNN baseline for return forecasting on stacked feature images.

    Expects input of shape (batch_size, 1, lookback, n_features).
    """

    def __init__(self, config: CnnConfig) -> None:
        super().__init__()
        self.config = config

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.conv_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn = nn.BatchNorm2d(config.conv_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Compute flattened size after conv + pool.
        h_out = config.lookback // 2
        w_out = config.n_features // 2
        flat_dim = config.conv_channels * h_out * w_out

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, lookback, n_features).
        """
        z = self.conv(x)
        z = self.bn(z)
        z = self.act(z)
        z = self.pool(z)
        z = torch.flatten(z, start_dim=1)
        out = self.fc(z)
        return out.squeeze(-1)

