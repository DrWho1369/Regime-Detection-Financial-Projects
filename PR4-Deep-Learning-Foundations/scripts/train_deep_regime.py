from __future__ import annotations

"""
Train a 2D CNN with DBE loss on stacked regime-aware features.

Supports both equity (Project 1) and gas (Project 1a) datasets by toggling
the --asset flag.
"""

from pathlib import Path
from typing import Literal, Tuple

import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from src.data.feature_stacker import (
    FeatureStackerConfig,
    build_stacked_frame,
)
from src.data.tensor_builder import build_cnn_tensors
from src.data.datasets import make_dataloaders
from src.models.cnn_baseline import CnnBaseline, CnnConfig, DirectionalBigErrorLoss


Asset = Literal["equity", "gas"]


def _split_by_date(
    dates: np.ndarray,
    train_end: str,
    val_end: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end_dt = np.datetime64(train_end)
    val_end_dt = np.datetime64(val_end)

    train_mask = dates <= train_end_dt
    val_mask = (dates > train_end_dt) & (dates <= val_end_dt)
    test_mask = dates > val_end_dt
    return train_mask, val_mask, test_mask


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN with DBE loss on stacked regime-aware features."
    )
    parser.add_argument(
        "--asset",
        type=str,
        choices=["equity", "gas"],
        default="equity",
        help="Which asset to train on (equity or gas).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback window size (image height).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Directional Big Error (DBE) penalty weight.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay for Adam.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2017-12-31",
        help="End date for the training period (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2018-12-31",
        help="End date for the validation period (YYYY-MM-DD).",
    )

    args = parser.parse_args()
    asset: Asset = args.asset  # type: ignore[assignment]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1) Build stacked frame (features + regime probs + returns)
    stack_cfg = FeatureStackerConfig()
    stacked = build_stacked_frame(asset=asset, config=stack_cfg)

    # 2) Build tensors
    feature_cols = [c for c in stacked.columns if not c.endswith("_log_return")]
    if asset == "equity":
        return_col = "target_log_return"
    else:
        return_col = "gas_log_return"

    X, y, dates = build_cnn_tensors(
        stacked,
        feature_cols=feature_cols,
        return_col=return_col,
        lookback=args.lookback,
    )

    dates_np = dates.to_numpy()
    train_mask, val_mask, test_mask = _split_by_date(
        dates_np, train_end=args.train_end, val_end=args.val_end
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_loader, val_loader = make_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=args.batch_size,
        shuffle_train=True,
    )

    # 3) Model + loss + optimiser
    n_features = X.shape[2]
    config = CnnConfig(
        n_features=n_features,
        lookback=args.lookback,
    )
    model = CnnBaseline(config).to(device)
    criterion = DirectionalBigErrorLoss(alpha=args.alpha)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 4) Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best val_loss={best_val_loss:.6f}."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 5) Final evaluation on test set
    if np.any(test_mask):
        test_ds = torch.utils.data.TensorDataset(
            torch.as_tensor(X_test, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(y_test, dtype=torch.float32),
        )
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test loss (DBE): {test_loss:.6f}")

    # 6) Save model weights and metadata
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if asset == "equity":
        model_path = models_dir / "cnn_v1_equity.pth"
    else:
        model_path = models_dir / "cnn_v1_gas.pth"

    torch.save(model.state_dict(), model_path)
    print(f"Saved best model state_dict to {model_path}")

    meta = {
        "asset": asset,
        "lookback": args.lookback,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "n_features": int(n_features),
        "best_val_loss": float(best_val_loss),
    }
    meta_path = models_dir / f"cnn_v1_{asset}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

