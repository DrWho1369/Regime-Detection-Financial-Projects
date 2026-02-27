from __future__ import annotations

"""
Run the trained CNN on the test set and create a regime-aware prediction plot.

This script:
- Rebuilds the stacked feature frame and CNN tensors.
- Loads the trained CNN weights for the chosen asset.
- Runs a forward pass on the test set to obtain predictions.
- Aligns predictions with HMM regimes from Project 3.
- Computes per-regime hit ratios (directional accuracy).
- Saves a 3-panel plot to the PR4 figures directory.
"""

from pathlib import Path
from typing import Literal, Tuple

import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.feature_stacker import FeatureStackerConfig, _load_csv, build_stacked_frame
from src.data.tensor_builder import build_cnn_tensors
from src.data.datasets import StackedWindowDataset
from src.models.cnn_baseline import CnnBaseline, CnnConfig
from src.visualization.prediction_plots import plot_regime_predictions


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CNN predictions against HMM regimes on the test set."
    )
    parser.add_argument(
        "--asset",
        type=str,
        choices=["equity", "gas"],
        default="equity",
        help="Which asset to analyse (equity or gas).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback window size used during training.",
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="DBE alpha used during training (for per-point loss diagnostics).",
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

    # 1) Rebuild stacked frame and tensors (must match training pipeline).
    stack_cfg = FeatureStackerConfig()
    stacked = build_stacked_frame(asset=asset, config=stack_cfg)

    feature_cols = [c for c in stacked.columns if not c.endswith("_log_return")]
    if asset == "equity":
        return_col = "target_log_return"
        hmm_probs_path = stack_cfg.equity_hmm_probs_path
    else:
        return_col = "gas_log_return"
        hmm_probs_path = stack_cfg.gas_hmm_probs_path

    X, y, dates = build_cnn_tensors(
        stacked,
        feature_cols=feature_cols,
        return_col=return_col,
        lookback=args.lookback,
    )

    dates_np = dates.to_numpy()
    _, _, test_mask = _split_by_date(
        dates_np, train_end=args.train_end, val_end=args.val_end
    )

    X_test, y_test, dates_test = X[test_mask], y[test_mask], dates[test_mask]
    if X_test.shape[0] == 0:
        raise ValueError("No test samples found for the given date split.")

    test_ds = StackedWindowDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # 2) Load trained CNN.
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    if asset == "equity":
        model_path = models_dir / "cnn_v1_equity.pth"
    else:
        model_path = models_dir / "cnn_v1_gas.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model weights not found at {model_path}. "
            f"Run train_deep_regime.py first."
        )

    n_features = X.shape[2]
    cfg = CnnConfig(n_features=n_features, lookback=args.lookback)
    model = CnnBaseline(cfg).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Forward pass to collect predictions.
    all_preds: list[float] = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy().ravel()
            all_preds.append(preds)

    y_pred = np.concatenate(all_preds)
    if y_pred.shape[0] != y_test.shape[0]:
        raise RuntimeError("Mismatch between number of predictions and test targets.")

    results_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        },
        index=dates_test,
    ).sort_index()

    # 4) Construct cumulative return index and error diagnostics.
    results_df["cum_return"] = (1.0 + results_df["y_true"]).cumprod()
    results_df["error"] = results_df["y_pred"] - results_df["y_true"]
    results_df["wrong_direction"] = np.sign(results_df["y_pred"]) != np.sign(
        results_df["y_true"]
    )

    # Optional: per-point DBE loss (not required for plotting but useful for inspection).
    sq_err = results_df["error"] ** 2
    penalty = 1.0 + args.alpha * results_df["wrong_direction"].astype(float)
    results_df["dbe_loss"] = sq_err * penalty

    # 5) Align with HMM regimes (Viterbi path) from Project 3.
    hmm_df = _load_csv(Path(hmm_probs_path))
    if "hmm_regime_id" not in hmm_df.columns:
        raise KeyError(
            f"'hmm_regime_id' column not found in HMM probs file at {hmm_probs_path}."
        )

    combined = results_df.join(hmm_df[["hmm_regime_id"]], how="inner")
    combined = combined.rename(columns={"hmm_regime_id": "regime"})
    if combined.empty:
        raise ValueError("No overlapping dates between CNN test set and HMM regimes.")

    # 6) Plot regime-aware predictions.
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"cnn_regime_predictions_{asset}.png"
    fig = plot_regime_predictions(combined, asset_name=asset, output_path=out_path)
    print(f"Saved regime-aware prediction plot to {out_path}")

    # 7) Directional accuracy (hit ratio) per regime.
    combined["correct_direction"] = (
        np.sign(combined["y_pred"]) == np.sign(combined["y_true"])
    ).astype(int)
    regime_stats = combined.groupby("regime")["correct_direction"].mean()

    print("\n--- Directional Accuracy (Hit Ratio) per Regime ---")
    for regime_id, hit_ratio in regime_stats.items():
        print(f"Regime {regime_id}: {hit_ratio:.2%}")


if __name__ == "__main__":
    main()

