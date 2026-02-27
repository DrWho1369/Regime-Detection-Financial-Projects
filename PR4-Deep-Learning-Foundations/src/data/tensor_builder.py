from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class TensorBuilderConfig:
    lookback: int = 10


def build_cnn_tensors(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    return_col: str,
    lookback: int = 10,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Transform a stacked daily panel into 3D tensors for CNN input.

    Each sample uses the previous ``lookback`` days of features to predict the
    1-day forward return at the window end.

    Parameters
    ----------
    df:
        Stacked feature frame indexed by date.
    feature_cols:
        Ordered list of feature column names to include.
    return_col:
        Name of the return column to use for the 1-day ahead target.
    lookback:
        Number of days in each window (image height).

    Returns
    -------
    X : np.ndarray
        Array of shape (samples, lookback, n_features).
    y : np.ndarray
        Array of shape (samples,) containing 1-day forward returns.
    dates : pd.DatetimeIndex
        Index of window end dates corresponding to each sample.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex for temporal ordering.")

    df = df.sort_index()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in stacked frame: {missing}")

    if return_col not in df.columns:
        raise KeyError(f"Return column {return_col!r} not found in stacked frame.")

    X_source = df[list(feature_cols)].to_numpy(dtype=float)
    y_forward = df[return_col].shift(-1)

    n_days = len(df)
    n_features = len(feature_cols)
    n_samples = n_days - lookback
    if n_samples <= 0:
        raise ValueError(
            f"Not enough observations ({n_days}) for lookback={lookback}."
        )

    X = np.zeros((n_samples, lookback, n_features), dtype=float)
    y = np.zeros(n_samples, dtype=float)
    dates: list[pd.Timestamp] = []

    y_array = y_forward.to_numpy()
    idx = df.index

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

    X = X[:j]
    y = y[:j]
    date_index = pd.DatetimeIndex(dates, name="end_date")
    return X, y, date_index

