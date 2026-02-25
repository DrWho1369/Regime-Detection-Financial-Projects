from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from ..config import END_DATE, START_DATE
from .io import save_dataframe
from ..utils.preprocess import forward_fill_panel


def fetch_raw_data(
    target_ticker: str = "^STOXX50E",
    start: str = START_DATE,
    end: str = END_DATE,
    auto_save: bool = True,
) -> pd.DataFrame:
    """
    Download and prepare raw daily data for the regime-detection project.

    This function:
    - Downloads Adjusted Close prices from Yahoo Finance for:
      * target equity index (default: ``^STOXX50E`` as SXXP proxy, or ``^GDAXI`` for DAX)
      * ``^VIX`` (VIX, used as a global fear proxy)
      * ``GC=F`` (Gold futures, proxy for PHAU)
      * ``BNDX`` (Vanguard Total International Bond ETF, proxy for Bunds / RX1)
    - Forward-fills missing values (holiday gaps) on the price series.
    - Computes daily log returns for the target index, Gold, and Bunds.
    - Saves the resulting panel to ``data/raw`` and returns it.

    Parameters
    ----------
    target_ticker:
        Yahoo Finance ticker for the target equity index (``^STOXX50E`` or ``^GDAXI``).
    start, end:
        Date range in ``YYYY-MM-DD`` format. Defaults to project-wide settings.
    auto_save:
        If True, writes the resulting DataFrame to ``data/raw``.
    """
    tickers: Dict[str, str] = {
        "target": target_ticker,
        "vstoxx": "^VIX",  # VIX as global volatility proxy
        "gold": "GC=F",
        "bund": "BNDX",
    }

    raw = yf.download(
        tickers=list(tickers.values()),
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for tickers {tickers} between {start} and {end}.")

    # yfinance returns a column MultiIndex when multiple tickers are requested.
    try:
        adj = raw["Adj Close"].copy()
    except KeyError as exc:
        raise KeyError("Expected 'Adj Close' in downloaded data.") from exc

    # Ensure a clean DateTime index and sort chronologically.
    adj.index = pd.to_datetime(adj.index)
    adj = adj.sort_index()
    adj.index.name = "date"

    # Map ticker symbols to logical names and reindex columns accordingly.
    inverse_map = {v: k for k, v in tickers.items()}
    adj = adj.rename(columns=inverse_map)

    # Forward-fill missing values (holiday data handling as in the paper).
    adj = forward_fill_panel(adj)

    # Compute daily log returns for target, Gold, and Bunds.
    price_cols_for_returns = ["target", "gold", "bund"]
    missing_return_cols = [c for c in price_cols_for_returns if c not in adj.columns]
    if missing_return_cols:
        raise KeyError(f"Missing expected price columns for returns: {missing_return_cols!r}")

    log_returns = np.log(adj[price_cols_for_returns]).diff()
    log_returns.columns = [f"{c}_log_return" for c in log_returns.columns]

    # Combine prices and returns into a single panel.
    panel = pd.concat([adj, log_returns], axis=1)

    if auto_save:
        # Name encodes the chosen target for clarity.
        target_label = inverse_map.get(target_ticker, "target")
        base_name = f"raw_panel_{target_label}_{start}_{end}"
        save_dataframe(panel, kind="raw", name=base_name, fmt="csv")

    return panel

