from __future__ import annotations

"""
Sentiment/macro feature definitions.

This module focuses on features derived from implied volatility indices such as
V2X/VDAX/VIX:
- Daily implied-volatility index changes (log returns).
- Vol-of-vol: rolling standard deviation of implied-vol log returns.
"""

from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["implied_vol_change"]


def implied_vol_change(
    prices: pd.Series,
    method: Literal["log_return", "simple_return", "diff"] = "log_return",
) -> pd.Series:
    """
    Compute daily implied-volatility index changes.

    Parameters
    ----------
    prices:
        Implied volatility index level series (e.g. V2X, VDAX, VIX).
    method:
        - ``\"log_return\"``: log(p_t / p_{t-1})
        - ``\"simple_return\"``: (p_t / p_{t-1}) - 1
        - ``\"diff\"``: p_t - p_{t-1}
    """
    if method == "log_return":
        change = np.log(prices).diff()
    elif method == "simple_return":
        change = prices.pct_change()
    elif method == "diff":
        change = prices.diff()
    else:
        raise ValueError(f"Unknown method: {method!r}")

    change.name = f"implied_vol_change_{method}"
    return change

