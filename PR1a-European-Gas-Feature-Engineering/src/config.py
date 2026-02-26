from __future__ import annotations

from pathlib import Path
from typing import Literal


# Base paths for Project 1a – European Gas Feature Engineering
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
FEATURES_DATA_DIR: Path = DATA_DIR / "features"


# Default date range (aligned with equity Project 1)
START_DATE: str = "2010-01-01"
END_DATE: str = "2020-12-31"


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
        One of "raw", "interim", or "features".
    name:
        Base file name without extension.
    fmt:
        File extension / format (e.g. "csv", "parquet").
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

