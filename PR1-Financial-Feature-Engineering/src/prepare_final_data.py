from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .features.feature_engineer import scale_features
from .visualize import create_feature_tensor


# Explicit list of the 14 feature columns used in the tensor representation.
# Adjust this list if you change the engineered feature set.
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


def load_feature_frame(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the engineered feature DataFrame from disk.

    By default expects ``data/processed/features.csv`` created earlier in the
    pipeline.
    """
    if path is None:
        path = DATA_DIR / "processed" / "features.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def standardize_features_time_safe(
    features: pd.DataFrame,
    train_end: str = "2017-12-31",
    return_col: str = "target_log_return",
    label_cols: Sequence[str] = ("y_forward_1d",),
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Apply StandardScaler to all feature columns in a time-safe manner.

    The scaler is fit **only** on observations up to ``train_end`` and then
    applied to the full 2010–2020 panel. Target/label columns are excluded
    from scaling.
    """
    # Use the explicit 14-feature list, ensuring all are present.
    missing = [c for c in FEATURE_COLUMNS_14 if c not in features.columns]
    if missing:
        raise KeyError(
            f"The following expected feature columns are missing from the input: {missing}. "
            "Make sure your engineered feature set matches FEATURE_COLUMNS_14."
        )
    feature_cols = [c for c in FEATURE_COLUMNS_14]

    scaled_all, _scaler = scale_features(
        features,
        train_end=train_end,
        feature_cols=feature_cols,
    )
    return scaled_all, feature_cols


def create_windows(
    data: pd.DataFrame,
    window_size: int = 10,
    feature_cols: Optional[Sequence[str]] = None,
    return_col: str = "target_log_return",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Wrapper around ``create_feature_tensor`` to build (N, window_size, n_features)
    tensors and 1-day forward targets from a feature DataFrame.
    """
    X, y, dates = create_feature_tensor(
        data,
        feature_cols=feature_cols,
        return_col=return_col,
        lookback=window_size,
    )
    return X, y, dates


def prepare_and_save_final_data(
    features_path: Optional[Path] = None,
    train_end: str = "2017-12-31",
    window_size: int = 10,
    return_col: str = "target_log_return",
) -> None:
    """
    Full pipeline:
    - Load engineered features.
    - Standardise features using training window (2010–2017).
    - Build 3D tensors of shape (samples, 10, 14) and aligned 1d-forward targets.
    - Split into train (<= train_end) and test (> train_end).
    - Save resulting arrays as .npy in ``data/processed``.
    """
    features = load_feature_frame(features_path)

    # Standardise using only pre-2018 data for fitting the scaler
    scaled_features, feature_cols = standardize_features_time_safe(
        features,
        train_end=train_end,
        return_col=return_col,
    )

    # Build windows and forward-return targets
    X_all, y_all, dates = create_windows(
        scaled_features,
        window_size=window_size,
        feature_cols=feature_cols,
        return_col=return_col,
    )

    # Train/test split in tensor space based on window end-date
    cutoff = pd.to_datetime(train_end)
    train_mask = dates <= cutoff
    test_mask = dates > cutoff

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)


if __name__ == "__main__":
    prepare_and_save_final_data()

