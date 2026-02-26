from __future__ import annotations

from pathlib import Path
from typing import Literal


# Base paths
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
FEATURES_DATA_DIR: Path = DATA_DIR / "features"


# Default date range for the project
START_DATE: str = "2010-01-01"
END_DATE: str = "2020-12-31"


# Logical names mapped to Yahoo Finance tickers
# These are easy to override if you prefer different symbols.
INDEX_TICKERS: dict[str, str] = {
    "sxxp": "^STOXX",  # STOXX Europe 600
    "dax": "^GDAXI",  # DAX Index
}

VOL_TICKERS: dict[str, str] = {
    "v2x": "^V2TX",  # Euro Stoxx 50 Volatility (VSTOXX)
    "vdax": "^VDAXI",  # VDAX-NEW
    "vix": "^VIX",  # CBOE Volatility Index
}


def get_data_path(
    kind: Literal["raw", "interim", "features"],
    name: str,
    fmt: str = "csv",
) -> Path:
    """
    Resolve a path within the data directory.

    Parameters
    ----------
    kind:
        One of ``\"raw\"``, ``\"interim\"``, or ``\"features\"``.
    name:
        Base file name without extension.
    fmt:
        File extension / format (e.g. ``\"csv\"``, ``\"parquet\"``).
    """
    if kind == "raw":
        base = RAW_DATA_DIR
    elif kind == "interim":
        base = INTERIM_DATA_DIR
    elif kind == "features":
        base = FEATURES_DATA_DIR
    else:
        raise ValueError(f"Unknown data kind: {kind!r}")

    return base / f"{name}.{fmt}"

