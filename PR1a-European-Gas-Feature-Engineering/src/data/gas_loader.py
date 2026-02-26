from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..config import END_DATE, START_DATE
from .io import save_dataframe


DEFAULT_TICKERS: Dict[str, str] = {
    "gas": "TTF=F",      # Dutch TTF Gas Futures (or configurable proxy)
    "brent": "BZ=F",     # Brent Oil Futures
    "carbon": "KRBN",    # EUA proxy (carbon ETF)
    "coal": "ICI=F",     # Placeholder for API2 coal; make configurable
}


def _download_prices(
    tickers: Dict[str, str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download Adjusted Close prices for the requested tickers."""
    raw = yf.download(
        tickers=list(tickers.values()),
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for tickers {tickers} between {start} and {end}.")

    try:
        adj = raw["Adj Close"].copy()
    except KeyError as exc:
        raise KeyError("Expected 'Adj Close' in downloaded data from yfinance.") from exc

    adj.index = pd.to_datetime(adj.index)
    adj = adj.sort_index()
    adj.index.name = "date"

    inverse_map = {v: k for k, v in tickers.items()}
    adj = adj.rename(columns=inverse_map)
    adj = adj.ffill()
    return adj


def _attach_log_returns(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute daily log returns for the given price columns and append them."""
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing expected price columns for log returns: {missing!r}")
    log_ret = np.log(panel[cols]).diff()
    log_ret.columns = [f"{c}_log_return" for c in cols]
    return pd.concat([panel, log_ret], axis=1)


def _attach_storage(
    panel: pd.DataFrame,
    storage_csv: Optional[Path],
) -> pd.DataFrame:
    """
    Merge European gas storage level and 5y seasonal average onto the panel.

    The CSV is expected to have at least:
      - 'date'
      - 'storage_level'
      - 'storage_5y_avg'
    """
    if storage_csv is None:
        return panel

    if not storage_csv.exists():
        raise FileNotFoundError(
            f"Gas storage CSV not found at {storage_csv}. "
            "Expected columns: 'date', 'storage_level', 'storage_5y_avg'."
        )

    storage = pd.read_csv(storage_csv, parse_dates=["date"]).set_index("date").sort_index()
    storage.index.name = "date"

    storage = storage.rename(
        columns={
            "storage_level": "gas_storage_level",
            "storage_5y_avg": "gas_storage_5y_avg",
        }
    )
    merged = panel.join(storage, how="left")
    # Storage deviation from 5-year average
    if {"gas_storage_level", "gas_storage_5y_avg"}.issubset(merged.columns):
        merged["gas_storage_dev"] = (
            merged["gas_storage_level"] - merged["gas_storage_5y_avg"]
        )
    return merged


def _attach_hdd(
    panel: pd.DataFrame,
    hdd_csv: Optional[Path],
    base_temp: float = 18.5,
) -> pd.DataFrame:
    """
    Merge Heating Degree Days (HDD) and HDD_change onto the panel.

    The CSV is expected to have:
      - 'date'
      - 'temp_avg' (daily mean temperature for NW Europe).
    """
    if hdd_csv is None:
        return panel

    if not hdd_csv.exists():
        raise FileNotFoundError(
            f"HDD CSV not found at {hdd_csv}. Expected columns: 'date', 'temp_avg'."
        )

    temp = pd.read_csv(hdd_csv, parse_dates=["date"]).set_index("date").sort_index()
    temp.index.name = "date"

    temp_avg = temp["temp_avg"]
    hdd = (base_temp - temp_avg).clip(lower=0.0)
    hdd.name = "hdd"
    hdd_change = hdd.diff()
    hdd_change.name = "hdd_change"

    hdd_df = pd.concat([hdd, hdd_change], axis=1)
    return panel.join(hdd_df, how="left")


def fetch_gas_panel(
    start: str = START_DATE,
    end: str = END_DATE,
    tickers: Optional[Dict[str, str]] = None,
    storage_csv: Optional[Path] = None,
    hdd_csv: Optional[Path] = None,
    auto_save: bool = True,
) -> pd.DataFrame:
    """
    Download and prepare the raw daily gas panel for Project 1a.

    This function:
      - Downloads Adjusted Close prices for:
          * gas (TTF futures or proxy)
          * brent (oil benchmark)
          * carbon (EUA proxy)
          * coal (API2 proxy)
      - Forward-fills missing prices and computes daily log returns.
      - Optionally merges:
          * gas storage level and 5y seasonal average (and deviation)
          * HDD and HDD_change (weather-driven demand factor)
      - Saves the resulting panel to data/raw/ and returns it.
    """
    tickers = tickers or DEFAULT_TICKERS

    prices = _download_prices(tickers, start=start, end=end)
    panel = _attach_log_returns(prices, cols=["gas", "brent", "carbon", "coal"])

    panel = _attach_storage(panel, storage_csv=storage_csv)
    panel = _attach_hdd(panel, hdd_csv=hdd_csv)

    if auto_save:
        base_name = f"raw_panel_gas_{start}_{end}"
        save_dataframe(panel, kind="raw", name=base_name, fmt="csv")

    return panel

