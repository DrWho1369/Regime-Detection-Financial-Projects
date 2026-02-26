from __future__ import annotations

"""
Momentum feature definitions.

Implements 3-month and 1-year momentum as cumulative log-returns over
the corresponding lookback windows.
"""

import numpy as np
import pandas as pd

__all__ = [
    "momentum",
    "momentum_3m",
    "momentum_12m",
]


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """
    Generic momentum: cumulative log-return over ``window`` trading days.
    """
    log_ret = np.log(prices).diff()
    # Lag the rolling sum by one day so that the momentum value at time t
    # only depends on returns up to t-1.
    mom = log_ret.rolling(window=window, min_periods=window).sum().shift(1)
    mom.name = f"momentum_{window}d"
    return mom


def momentum_3m(prices: pd.Series) -> pd.Series:
    """
    3-month momentum (approx. 63 trading days).
    """
    return momentum(prices, window=63)


def momentum_12m(prices: pd.Series) -> pd.Series:
    """
    1-year momentum (approx. 252 trading days).
    """
    return momentum(prices, window=252)

