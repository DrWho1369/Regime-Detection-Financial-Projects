from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd
import yfinance as yf

from ..config import END_DATE, INDEX_TICKERS, START_DATE, VOL_TICKERS
from .io import save_dataframe


def _clean_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise a yfinance OHLCV frame to a daily close-price series.
    """
    if raw.empty:
        raise ValueError("Received empty DataFrame from data provider.")

    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(how="all")

    if "Close" not in df.columns:
        raise KeyError("Expected a 'Close' column in downloaded data.")

    return df[["Close"]].rename(columns={"Close": "close"})


def fetch_index_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    auto_save: bool = True,
    save_name: str | None = None,
) -> pd.DataFrame:
    """
    Download daily index prices from Yahoo Finance and return a close-price series.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. ``\"^STOXX\"``, ``\"^GDAXI\"``).
    start, end:
        Date range strings in ``YYYY-MM-DD`` format. Defaults to the
        project-level ``START_DATE`` / ``END_DATE`` when omitted.
    auto_save:
        If True, persist the cleaned close-price series to ``data/raw``.
    save_name:
        Optional base filename (without extension) to use when saving. If
        omitted, derives a name from the ticker symbol.
    """
    start_ = start or START_DATE
    end_ = end or END_DATE

    raw = yf.download(ticker, start=start_, end=end_, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker {ticker} between {start_} and {end_}.")

    df = _clean_price_frame(raw)

    if auto_save:
        base_name = save_name or f"index_{ticker.replace('^', '').lower()}_prices"
        save_dataframe(df, kind="raw", name=base_name, fmt="csv")

    return df


def fetch_named_indices(
    names: Sequence[str] = ("sxxp", "dax"),
    start: str | None = None,
    end: str | None = None,
    auto_save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch one or more indices by logical name as defined in ``INDEX_TICKERS``.

    Returns a mapping from logical name (e.g. ``\"sxxp\"``, ``\"dax\"``)
    to close-price DataFrames.
    """
    missing = [name for name in names if name not in INDEX_TICKERS]
    if missing:
        raise KeyError(f"Unknown index logical name(s): {missing!r}")

    out: Dict[str, pd.DataFrame] = {}
    for name in names:
        ticker = INDEX_TICKERS[name]
        df = fetch_index_prices(
            ticker=ticker,
            start=start,
            end=end,
            auto_save=auto_save,
            save_name=f"index_{name}_prices",
        )
        out[name] = df
    return out


def fetch_vol_indices(
    names: Iterable[str] = ("v2x", "vdax", "vix"),
    start: str | None = None,
    end: str | None = None,
    auto_save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch implied volatility indices (e.g. V2X/VDAX/VIX) by logical name.

    Returns a mapping from logical name to close-price DataFrames.
    """
    available: Mapping[str, str] = VOL_TICKERS
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"Unknown vol index logical name(s): {missing!r}")

    out: Dict[str, pd.DataFrame] = {}
    for name in names:
        ticker = available[name]
        df = fetch_index_prices(
            ticker=ticker,
            start=start,
            end=end,
            auto_save=auto_save,
            save_name=f"vol_{name}_index",
        )
        out[name] = df
    return out

