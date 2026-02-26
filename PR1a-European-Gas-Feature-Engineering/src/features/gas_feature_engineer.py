from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import talib


TRADING_DAYS_PER_YEAR = 252

# Canonical 14-feature schema (names aligned with equity Project 1)
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


def shannon_entropy_binary(returns: pd.Series, window: int = 66) -> pd.Series:
    s = (returns > 0).astype(int)
    p_up = s.rolling(window=window, min_periods=window).mean().shift(1)
    p_down = 1.0 - p_up
    eps = 1e-10
    p_up_safe = p_up.clip(eps, 1.0 - eps)
    p_down_safe = p_down.clip(eps, 1.0 - eps)
    entropy = -(p_up_safe * np.log(p_up_safe) + p_down_safe * np.log(p_down_safe))
    entropy.name = f"entropy_{window}d"
    return entropy


def vol_scaled_returns(returns: pd.Series, vol_window: int = 126) -> pd.Series:
    rolling_vol = (
        returns.rolling(window=vol_window, min_periods=vol_window).std().shift(1)
    )
    scaled = returns / rolling_vol
    scaled.name = f"vol_scaled_return_{vol_window}d"
    return scaled


def rolling_skewness(returns: pd.Series, window: int = 126) -> pd.Series:
    skew = returns.rolling(window=window, min_periods=window).skew().shift(1)
    skew.name = f"skewness_{window}d"
    return skew


def technical_indicators(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.DataFrame:
    if high is None:
        high = close
    if low is None:
        low = close

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
    bollinger_pct_b.name = "bollinger_pct_b_20d"

    cci_values = talib.CCI(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        timeperiod=20,
    )
    cmci = pd.Series(cci_values, index=close.index, name="cmci_20d")

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

    rsi_values = talib.RSI(close.values.astype(float), timeperiod=14)
    rsi = pd.Series(rsi_values, index=close.index, name="rsi_14d")

    willr_values = talib.WILLR(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        timeperiod=14,
    )
    willr = pd.Series(willr_values, index=close.index, name="williams_r_14d")

    tech = pd.concat([bollinger_pct_b, cmci, stoch_k, rsi, willr], axis=1)
    return tech


def _momentum(prices: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(prices).diff()
    mom = log_ret.rolling(window=window, min_periods=window).sum().shift(1)
    mom.name = f"momentum_{window}d"
    return mom


def momentum_3m(prices: pd.Series) -> pd.Series:
    return _momentum(prices, window=63)


def momentum_12m(prices: pd.Series) -> pd.Series:
    return _momentum(prices, window=252)


def build_gas_feature_panel(
    data: pd.DataFrame,
    drop_initial_year: bool = True,
) -> pd.DataFrame:
    """
    Build the 14-feature gas panel, mirroring the equity feature set but with
    gas-specific proxies:

    - Statistical, technical, momentum features computed on gas prices/returns.
    - vstoxx_chg := coal_log_return (API2 coal proxy).
    - vol_of_vol := gas_storage_dev (deviation from 5y seasonal average).
    - gold_log_return := brent_log_return.
    - bund_rets := carbon_log_return.
    """
    required_cols = [
        "gas",
        "gas_log_return",
        "brent_log_return",
        "carbon_log_return",
        "coal_log_return",
    ]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns in gas panel: {missing!r}")

    gas_ret = data["gas_log_return"]
    gas_price = data["gas"]

    entropy = shannon_entropy_binary(gas_ret, window=66)
    scaled_returns = vol_scaled_returns(gas_ret, vol_window=126)
    skew = rolling_skewness(gas_ret, window=126)

    tech = technical_indicators(close=gas_price)

    m3 = momentum_3m(gas_price).rename("momentum_63d")
    m12 = momentum_12m(gas_price).rename("momentum_252d")

    if "gas_storage_dev" in data.columns:
        storage_dev = data["gas_storage_dev"]
    else:
        storage_dev = pd.Series(index=data.index, dtype=float)
    vol_of_vol = storage_dev.rename("vol_of_vol")

    vstoxx_chg = data["coal_log_return"].rename("vstoxx_chg")
    gold_log_return = data["brent_log_return"].rename("gold_log_return")
    bund_rets = data["carbon_log_return"].rename("bund_rets")

    pieces = [
        entropy,
        scaled_returns,
        skew,
        vol_of_vol,
        tech["bollinger_pct_b_20d"],
        tech["cmci_20d"],
        tech["stoch_k_14d"],
        tech["rsi_14d"],
        tech["williams_r_14d"],
        m3,
        m12,
        vstoxx_chg,
        gold_log_return,
        bund_rets,
    ]

    features = pd.concat(pieces, axis=1)

    # Optional: keep HDD and storage level/dev as extra exogenous columns
    extra_cols = []
    for col in ("hdd", "hdd_change", "gas_storage_level", "gas_storage_5y_avg", "gas_storage_dev"):
        if col in data.columns:
            extra_cols.append(data[col])
    if extra_cols:
        features = pd.concat([features] + extra_cols, axis=1)

    if drop_initial_year and len(features) > TRADING_DAYS_PER_YEAR:
        features = features.iloc[TRADING_DAYS_PER_YEAR:]

    # Ensure consistent ordering of the canonical 14-feature set
    features = features.dropna(subset=list(FEATURE_COLUMNS_14), how="any")
    features = features.loc[:, [c for c in FEATURE_COLUMNS_14 if c in features.columns]] + features.drop(
        columns=[c for c in FEATURE_COLUMNS_14 if c in features.columns]
    )
    return features

