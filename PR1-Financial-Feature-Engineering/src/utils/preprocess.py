from __future__ import annotations

"""
Preprocessing utilities shared across the project.

Includes:
- Forward-fill helpers for handling missing observations.
- Time-safe feature normalisation helpers.
"""

from typing import Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def forward_fill_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing values in a panel DataFrame.
    """
    return df.ffill()


def align_and_forward_fill(series_map: Mapping[str, pd.Series]) -> pd.DataFrame:
    """
    Align multiple time series on a common index and forward-fill gaps.
    """
    df = pd.concat(series_map, axis=1)
    df.columns = list(series_map.keys())
    return df.ffill()


def time_safe_standardize(
    features: pd.DataFrame,
    train_end: str,
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise features using only data up to ``train_end`` to fit the scaler.
    """
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

    return scaled, scaler

