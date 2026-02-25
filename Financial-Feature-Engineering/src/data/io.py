from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from ..config import FEATURES_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR


DataKind = Literal["raw", "interim", "features"]


def _kind_to_dir(kind: DataKind) -> Path:
    if kind == "raw":
        return RAW_DATA_DIR
    if kind == "interim":
        return INTERIM_DATA_DIR
    if kind == "features":
        return FEATURES_DATA_DIR
    raise ValueError(f"Unknown data kind: {kind!r}")


def save_dataframe(
    df: pd.DataFrame,
    kind: DataKind,
    name: str,
    fmt: str = "csv",
    index: bool = True,
    path_override: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Save a DataFrame to disk under the standard data directory structure.

    Parameters
    ----------
    df:
        DataFrame to save.
    kind:
        One of ``\"raw\"``, ``\"interim\"``, or ``\"features\"``.
    name:
        Base filename without extension.
    fmt:
        File format: ``\"csv\"`` or ``\"parquet\"`` (default: ``\"csv\"``).
    index:
        Whether to write the index to disk (default: True).
    path_override:
        Optional explicit path to use instead of the standard location.
    kwargs:
        Passed through to the underlying pandas IO function.
    """
    if path_override is not None:
        path = path_override
    else:
        base_dir = _kind_to_dir(kind)
        path = base_dir / f"{name}.{fmt}"

    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        if "date_format" not in kwargs:
            # Encourage ISO-8601 representation for dates
            kwargs.setdefault("date_format", "%Y-%m-%d")
        df.to_csv(path, index=index, **kwargs)
    elif fmt == "parquet":
        df.to_parquet(path, index=index, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")

    return path


def load_dataframe(
    kind: DataKind,
    name: str,
    fmt: str = "csv",
    path_override: Optional[Path] = None,
    parse_dates: bool = True,
    index_col: int | str | None = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a DataFrame from the standard data directory structure.

    Parameters
    ----------
    kind:
        One of ``\"raw\"``, ``\"interim\"``, or ``\"features\"``.
    name:
        Base filename without extension.
    fmt:
        File format: ``\"csv\"`` or ``\"parquet\"`` (default: ``\"csv\"``).
    path_override:
        Optional explicit path to use instead of the standard location.
    parse_dates:
        If True, attempt to parse dates when reading CSV.
    index_col:
        Column to use as index when reading CSV (default: first column).
    kwargs:
        Passed through to the underlying pandas IO function.
    """
    if path_override is not None:
        path = path_override
    else:
        base_dir = _kind_to_dir(kind)
        path = base_dir / f"{name}.{fmt}"

    if not path.exists():
        raise FileNotFoundError(path)

    if fmt == "csv":
        if parse_dates:
            kwargs.setdefault("parse_dates", True)
        if index_col is not None:
            kwargs.setdefault("index_col", index_col)
        return pd.read_csv(path, **kwargs)
    if fmt == "parquet":
        return pd.read_parquet(path, **kwargs)

    raise ValueError(f"Unsupported format: {fmt!r}")

