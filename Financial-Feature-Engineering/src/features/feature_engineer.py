from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler


TRADING_DAYS_PER_YEAR = 252


def shannon_entropy_binary(
    returns: pd.Series,
    window: int = 66,
) -> pd.Series:
    """
    Rolling Shannon entropy on the sign of returns over a fixed window.

    S_t = 1 if r_t > 0 else 0
    H_t = - sum_i p_i log(p_i), i in {0, 1}
    """
    # Binary sequence: 1 for up day, 0 otherwise (flat and down treated the same).
    s = (returns > 0).astype(int)

    # Rolling mean of the binary sequence is P(S=1), lagged by one day
    # to ensure the feature at time t only uses information up to t-1.
    p_up = s.rolling(window=window, min_periods=window).mean().shift(1)
    p_down = 1.0 - p_up

    # Avoid log(0) by clipping probabilities.
    eps = 1e-10
    p_up_safe = p_up.clip(eps, 1.0 - eps)
    p_down_safe = p_down.clip(eps, 1.0 - eps)

    entropy = -(p_up_safe * np.log(p_up_safe) + p_down_safe * np.log(p_down_safe))
    entropy.name = f"entropy_{window}d"
    return entropy


def vol_scaled_returns(
    returns: pd.Series,
    vol_window: int = 126,
) -> pd.Series:
    """
    Volatility-scaled returns using a rolling standard deviation.

    scaled_t = r_t / sigma_t
    where sigma_t is the rolling std of returns over ``vol_window`` days.
    """
    rolling_vol = returns.rolling(window=vol_window, min_periods=vol_window).std().shift(1)
    scaled = returns / rolling_vol
    scaled.name = f"vol_scaled_return_{vol_window}d"
    return scaled


def rolling_skewness(
    returns: pd.Series,
    window: int = 126,
) -> pd.Series:
    """
    Rolling skewness of returns over the specified window.
    """
    skew = returns.rolling(window=window, min_periods=window).skew().shift(1)
    skew.name = f"skewness_{window}d"
    return skew


def vol_of_vol(
    vstoxx_prices: pd.Series,
    window: int = 66,
) -> pd.Series:
    """
    Vol-of-vol: rolling std of VSTOXX log returns over ``window`` days.
    """
    log_ret = np.log(vstoxx_prices).diff()
    vol = log_ret.rolling(window=window, min_periods=window).std().shift(1)
    vol.name = f"vstoxx_vol_of_vol_{window}d"
    return vol


def technical_indicators(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute technical indicators using TA-Lib:
    - Bollinger %B (20-day)
    - CMCI (implemented via CCI, 20-day)
    - Stochastic %K (14-day)
    - RSI (14-day)
    - Williams %R (14-day)

    If high/low are not provided, they are approximated with ``close``.
    """
    if high is None:
        high = close
    if low is None:
        low = close

    # Bollinger Bands %B (20-day, 2 std dev)
    upper, middle, lower = talib.BBANDS(
        close.values.astype(float),
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0,
    )
    bb_upper = pd.Series(upper, index=close.index)
    bb_lower = pd.Series(lower, index=close.index)
    bollinger_pct_b = (close - bb_lower) / (bb_upper - bb_lower)

    # CMCI via CCI over 20 days
    cci_values = talib.CCI(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        timeperiod=20,
    )
    cmci = pd.Series(cci_values, index=close.index, name="cmci_20d")

    # Stochastic %K (14-day)
    slowk, _slowd = talib.STOCH(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    stoch_k = pd.Series(slowk, index=close.index, name="stoch_k_14d")

    # RSI (14-day)
    rsi_values = talib.RSI(close.values.astype(float), timeperiod=14)
    rsi = pd.Series(rsi_values, index=close.index, name="rsi_14d")

    # Williams %R (14-day)
    willr_values = talib.WILLR(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        timeperiod=14,
    )
    willr = pd.Series(willr_values, index=close.index, name="williams_r_14d")

    bollinger_pct_b.name = "bollinger_pct_b_20d"

    tech = pd.concat([bollinger_pct_b, cmci, stoch_k, rsi, willr], axis=1)
    return tech


def build_feature_panel(
    data: pd.DataFrame,
    target_return_col: str = "target_log_return",
    target_price_col: str = "target",
    vstoxx_price_col: str = "vstoxx",
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    drop_initial_year: bool = True,
) -> pd.DataFrame:
    """
    Build the main statistical and technical feature panel from a raw data frame.

    The input ``data`` is expected to contain at least:
    - target price series (``target_price_col``)
    - target log returns (``target_return_col``)
    - VSTOXX price series (``vstoxx_price_col``)

    Optionally, high/low series for technical indicators.
    """
    if target_return_col not in data.columns:
        raise KeyError(f"Missing target return column: {target_return_col!r}")
    if target_price_col not in data.columns:
        raise KeyError(f"Missing target price column: {target_price_col!r}")
    if vstoxx_price_col not in data.columns:
        raise KeyError(f"Missing VSTOXX price column: {vstoxx_price_col!r}")

    target_ret = data[target_return_col]
    target_price = data[target_price_col]
    vstoxx_price = data[vstoxx_price_col]

    entropy = shannon_entropy_binary(target_ret, window=66)
    scaled_returns = vol_scaled_returns(target_ret, vol_window=126)
    skew = rolling_skewness(target_ret, window=126)
    vov = vol_of_vol(vstoxx_price, window=66)

    if high_col is not None and low_col is not None:
        tech = technical_indicators(
            close=target_price,
            high=data[high_col],
            low=data[low_col],
        )
    else:
        tech = technical_indicators(close=target_price)

    features = pd.concat(
        [entropy, scaled_returns, skew, vov, tech],
        axis=1,
    )

    if drop_initial_year and len(features) > TRADING_DAYS_PER_YEAR:
        features = features.iloc[TRADING_DAYS_PER_YEAR:]

    return features.dropna(how="all")


def scale_features(
    features: pd.DataFrame,
    train_end: pd.Timestamp | str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Time-safe standardisation of features using scikit-learn's StandardScaler.

    The scaler is fit **only** on the training period (up to and including
    ``train_end``) to avoid look-ahead bias, then applied to the full panel.
    """
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must have a DatetimeIndex for time-based splitting.")

    if feature_cols is None:
        feature_cols = list(features.columns)

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

