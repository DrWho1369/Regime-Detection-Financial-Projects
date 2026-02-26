from __future__ import annotations

"""
Technical indicator feature definitions.

Wraps TA-Lib to provide:
- Bollinger %B
- CMCI (via CCI)
- Stochastic %K
- RSI
- Williams %R
"""

from typing import Optional

import pandas as pd
import talib

__all__ = ["technical_indicators"]


def technical_indicators(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute technical indicators using TA-Lib.

    If ``high``/``low`` are not provided, they are approximated with ``close``.
    """
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

