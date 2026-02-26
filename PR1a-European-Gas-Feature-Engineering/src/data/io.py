from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get_data_path


def save_dataframe(
    df: pd.DataFrame,
    kind: str,
    name: str,
    fmt: str = "csv",
) -> Path:
    """
    Save a DataFrame into the Project 1a data directory.

    Parameters
    ----------
    df:
        DataFrame to write.
    kind:
        One of "raw", "interim", or "features".
    name:
        Base file name without extension.
    fmt:
        File extension / format (default: "csv").
    """
    path = get_data_path(kind=kind, name=name, fmt=fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=True, date_format="%Y-%m-%d")
    else:
        raise ValueError(f"Unsupported format for save_dataframe: {fmt!r}")
    return path

