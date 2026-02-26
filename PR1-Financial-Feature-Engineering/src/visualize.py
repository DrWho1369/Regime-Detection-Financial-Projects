from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DATA_DIR


def plot_correlation_heatmap(
    features: pd.DataFrame,
    target_return_col: str = "target_log_return",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Plot and optionally save a correlation heatmap of features + target return.

    Returns the correlation matrix for further inspection.
    """
    df = features.copy()
    if target_return_col in df.columns:
        corr_input = df
    else:
        raise KeyError(f"{target_return_col!r} must be present in the input for correlation.")

    corr = corr_input.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.close()

    return corr


def create_feature_tensor(
    features: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    return_col: str = "target_log_return",
    lookback: int = 10,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Transform a (N_days, n_features) DataFrame into a 3D tensor of shape
    (samples, lookback, n_features) plus a target vector of 1-day ahead returns.

    Each sample uses the previous ``lookback`` days of features to predict the
    1-day forward log return of the index.
    """
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must have a DatetimeIndex for temporal ordering.")

    features = features.sort_index()

    if feature_cols is None:
        # Exclude the return column from the feature stack if present
        feature_cols = [c for c in features.columns if c != return_col]

    if return_col not in features.columns:
        raise KeyError(f"Return column {return_col!r} not found in features.")

    X_source = features[feature_cols].to_numpy(dtype=float)
    y_forward = features[return_col].shift(-1)

    n_days = len(features)
    n_features = len(feature_cols)
    n_samples = n_days - lookback
    if n_samples <= 0:
        raise ValueError("Not enough observations to build the requested lookback tensor.")

    X = np.zeros((n_samples, lookback, n_features), dtype=float)
    y = np.zeros(n_samples, dtype=float)
    dates: list[pd.Timestamp] = []

    y_array = y_forward.to_numpy()
    idx = features.index

    j = 0
    for end_idx in range(lookback - 1, n_days - 1):
        y_val = y_array[end_idx]
        if np.isnan(y_val):
            continue
        window_start = end_idx - lookback + 1
        window_end = end_idx + 1
        X[j] = X_source[window_start:window_end]
        y[j] = y_val
        dates.append(idx[end_idx])
        j += 1

    # Trim in case we skipped NaNs
    X = X[:j]
    y = y[:j]
    date_index = pd.DatetimeIndex(dates, name="end_date")

    return X, y, date_index


def add_forward_return_label(
    features: pd.DataFrame,
    return_col: str = "target_log_return",
    label_name: str = "y_forward_1d",
) -> pd.DataFrame:
    """
    Attach a 1-day forward log-return label to the feature DataFrame.
    """
    if return_col not in features.columns:
        raise KeyError(f"Return column {return_col!r} not found in features.")

    out = features.copy()
    out[label_name] = out[return_col].shift(-1)
    return out


def plot_entropy_vs_price(
    price: pd.Series,
    entropy: pd.Series,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot entropy over time alongside the target index price.

    Includes vertical annotations for key events such as:
    - 2015-01-22 (ECB QE announcement)
    - 2020-03-12 (COVID-19 shock in Europe)
    """
    if not isinstance(price.index, pd.DatetimeIndex) or not isinstance(
        entropy.index, pd.DatetimeIndex
    ):
        raise TypeError("price and entropy must both have DatetimeIndex indexes.")

    aligned = pd.concat([price.rename("price"), entropy.rename("entropy")], axis=1).dropna()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(aligned.index, aligned["price"], color="tab:blue", label="Index Price")
    ax1.set_ylabel("Price", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(aligned.index, aligned["entropy"], color="tab:red", label="Entropy (66d)")
    ax2.set_ylabel("Entropy", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Annotate major macro/market shifts
    events = {
        "2015-01-22": "ECB QE",
        "2020-03-12": "COVID shock",
    }
    for date_str, label in events.items():
        date = pd.to_datetime(date_str)
        if aligned.index.min() <= date <= aligned.index.max():
            ax1.axvline(date, color="grey", linestyle="--", alpha=0.7)
            ax1.text(
                date,
                ax1.get_ylim()[1],
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="grey",
            )

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_processed_features(
    features: pd.DataFrame,
    filename: str = "features.csv",
) -> Path:
    """
    Save the final engineered feature set for downstream models.

    Defaults to ``data/processed/features.csv`` to separate it from raw and
    intermediate artifacts.
    """
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / filename
    features.to_csv(path, index=True, date_format="%Y-%m-%d")
    return path

