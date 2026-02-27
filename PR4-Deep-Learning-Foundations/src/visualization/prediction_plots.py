from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_regime_spans(dates: pd.DatetimeIndex, regimes: pd.Series) -> list[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Compute contiguous spans where the regime label is constant.
    """
    spans: list[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if len(dates) == 0:
        return spans

    current_regime = int(regimes.iloc[0])
    start_idx = 0
    for i in range(1, len(dates)):
        r = int(regimes.iloc[i])
        if r != current_regime:
            spans.append((dates[start_idx], dates[i - 1], current_regime))
            current_regime = r
            start_idx = i
    spans.append((dates[start_idx], dates[-1], current_regime))
    return spans


def plot_regime_predictions(
    results_df: pd.DataFrame,
    asset_name: str,
    output_path: Path | str,
) -> plt.Figure:
    """
    Plot CNN predictions and errors against HMM regimes.

    Expects ``results_df`` indexed by date, with at least:
    - ``cum_return``: cumulative return index (e.g. (1 + r).cumprod()).
    - ``y_true``: realised returns.
    - ``y_pred``: CNN-predicted returns.
    - ``error``: prediction error (y_pred - y_true).
    - ``wrong_direction``: boolean indicating sign mismatch.
    - ``regime``: integer HMM regime id (Viterbi path).
    """
    if not isinstance(results_df.index, pd.DatetimeIndex):
        raise TypeError("results_df must have a DatetimeIndex.")

    required_cols = {
        "cum_return",
        "y_true",
        "y_pred",
        "wrong_direction",
        "regime",
    }
    missing = required_cols.difference(results_df.columns)
    if missing:
        raise KeyError(f"results_df is missing required columns: {sorted(missing)}")

    dates = results_df.index
    regimes = results_df["regime"]

    fig, (ax_price, ax_pred, ax_err) = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1]},
    )

    # Top panel: cumulative return, shaded by regime.
    ax_price.plot(dates, results_df["cum_return"], color="black", label="Cumulative return index")

    spans = _get_regime_spans(dates, regimes)
    cmap = plt.get_cmap("tab10")
    y_min, y_max = np.nanmin(results_df["cum_return"].values), np.nanmax(
        results_df["cum_return"].values
    )
    for start, end, reg in spans:
        color = cmap(reg % 10)
        ax_price.axvspan(start, end, color=color, alpha=0.15)

    ax_price.set_ylabel("Cumulative\nreturn")
    ax_price.set_title(f"{asset_name.upper()} – CNN predictions vs HMM regimes")
    ax_price.legend(loc="upper left")

    # Middle panel: predicted vs actual returns.
    ax_pred.plot(
        dates,
        results_df["y_pred"],
        label="Predicted return",
        color="tab:orange",
        linewidth=1.0,
    )
    ax_pred.scatter(
        dates,
        results_df["y_true"],
        label="Actual return",
        color="tab:blue",
        s=8,
        alpha=0.4,
    )
    ax_pred.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax_pred.set_ylabel("Return")
    ax_pred.legend(loc="upper left")

    # Bottom panel: cumulative count of sign mismatches (directional errors).
    wrong_mask = results_df["wrong_direction"].astype(bool)
    cum_mismatch = wrong_mask.cumsum()
    ax_err.plot(
        dates,
        cum_mismatch,
        color="tab:red",
        linewidth=1.0,
        label="Cumulative sign mismatches",
    )
    ax_err.set_ylabel("Cumulative\nmismatches")
    ax_err.set_xlabel("Date")
    ax_err.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig

