from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR


FEATURE_COLUMNS_14: Sequence[str] = (
    "entropy_66d",
    "vol_scaled_return_126d",
    "skewness_126d",
    "vol_of_vol",
    "bollinger_pct_b_20d",
    "cmci_20d",
    "stoch_k_14d",
    "rsi_14d",
    "williams_r_14d",
    "momentum_63d",
    "momentum_252d",
    "vstoxx_chg",
    "gold_log_return",
    "bund_rets",
)


def load_gas_feature_frame(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the engineered gas feature DataFrame from disk.

    By default expects data/processed/gas_features.csv within Project 1a.
    Ensures the index is a DatetimeIndex for time-based splitting.
    """
    if path is None:
        path = DATA_DIR / "processed" / "gas_features.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def standardize_gas_features_time_safe(
    features: pd.DataFrame,
    train_end: str = "2017-12-31",
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Apply StandardScaler to the 14 canonical gas feature columns in a time-safe way.

    The scaler is fit only on observations up to train_end and then applied
    to the full 2010–2020 panel.
    """
    missing = [c for c in FEATURE_COLUMNS_14 if c not in features.columns]
    if missing:
        raise KeyError(
            f"The following expected feature columns are missing from the input: {missing}. "
            "Ensure gas_features.csv matches the canonical FEATURE_COLUMNS_14 schema."
        )

    feature_cols = list(FEATURE_COLUMNS_14)
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must have a DatetimeIndex for time-based splitting.")

    train_mask = features.index <= pd.to_datetime(train_end)
    train_data = features.loc[train_mask, feature_cols]
    if train_data.isna().any().any():
        train_data = train_data.dropna()

    scaler = StandardScaler()
    scaler.fit(train_data.values)

    scaled_values = scaler.transform(features[feature_cols].values)
    scaled = features.copy()
    scaled.loc[:, feature_cols] = scaled_values
    return scaled, feature_cols


def create_windows(
    data: pd.DataFrame,
    window_size: int = 10,
    feature_cols: Optional[Sequence[str]] = None,
    return_col: str = "gas_log_return",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build (N, window_size, n_features) tensors and 1-day forward gas returns
    from a feature DataFrame, mirroring Project 1's tensor creation logic.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex for temporal ordering.")

    data = data.sort_index()
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != return_col]

    if return_col not in data.columns:
        raise KeyError(f"Return column {return_col!r} not found in features.")

    X_source = data[feature_cols].to_numpy(dtype=float)
    y_forward = data[return_col].shift(-1)

    n_days = len(data)
    n_features = len(feature_cols)
    n_samples = n_days - window_size
    if n_samples <= 0:
        raise ValueError("Not enough observations to build the requested lookback tensor.")

    X = np.zeros((n_samples, window_size, n_features), dtype=float)
    y = np.zeros(n_samples, dtype=float)
    dates: list[pd.Timestamp] = []

    y_array = y_forward.to_numpy()
    idx = data.index

    j = 0
    for end_idx in range(window_size - 1, n_days - 1):
        y_val = y_array[end_idx]
        if np.isnan(y_val):
            continue
        window_start = end_idx - window_size + 1
        window_end = end_idx + 1
        X[j] = X_source[window_start:window_end]
        y[j] = y_val
        dates.append(idx[end_idx])
        j += 1

    X = X[:j]
    y = y[:j]
    date_index = pd.DatetimeIndex(dates, name="end_date")
    return X, y, date_index


def prepare_and_save_gas_data(
    features_path: Optional[Path] = None,
    train_end: str = "2017-12-31",
    window_size: int = 10,
    return_col: str = "gas_log_return",
) -> None:
    """
    Full gas pipeline:
      - Load engineered gas features from gas_features.csv.
      - Standardise the 14 canonical features using training window (2010–2017).
      - Build 3D tensors of shape (samples, window_size, 14) and 1d-forward gas targets.
      - Split into train (<= train_end) and test (> train_end).
      - Save resulting arrays as .npy in data/processed (Project 1a).
    """
    features = load_gas_feature_frame(features_path)
    scaled_features, feature_cols = standardize_gas_features_time_safe(
        features, train_end=train_end
    )

    X_all, y_all, dates = create_windows(
        scaled_features,
        window_size=window_size,
        feature_cols=feature_cols,
        return_col=return_col,
    )

    cutoff = pd.to_datetime(train_end)
    train_mask = dates <= cutoff
    test_mask = dates > cutoff

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_gas_train.npy", X_train)
    np.save(processed_dir / "y_gas_train.npy", y_train)
    np.save(processed_dir / "X_gas_test.npy", X_test)
    np.save(processed_dir / "y_gas_test.npy", y_test)


if __name__ == "__main__":
    prepare_and_save_gas_data()

