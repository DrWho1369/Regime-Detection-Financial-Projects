from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_gmm_vs_hmm_price(
    raw_price_path: Path,
    gmm_probs_path: Path,
    hmm_probs_path: Path,
    output_path: Path,
    price_col: str = "target",
    gmm_regime_col: str = "regime_id",
    hmm_regime_col: str = "hmm_regime_id",
) -> None:
    """
    Compare GMM and HMM regimes on the same price series in a two-panel chart.

    Top panel: price with GMM regimes shaded.
    Bottom panel: price with HMM regimes shaded.
    """
    price_panel = (
        pd.read_csv(raw_price_path, index_col=0, parse_dates=True)
        .sort_index()
    )
    prices = price_panel[price_col].rename("price")

    gmm = pd.read_csv(gmm_probs_path, index_col=0, parse_dates=True).sort_index()
    hmm = pd.read_csv(hmm_probs_path, index_col=0, parse_dates=True).sort_index()

    # Align on common dates
    df = pd.concat(
        [
            prices,
            gmm[[gmm_regime_col]].rename(columns={gmm_regime_col: "gmm_regime_id"}),
            hmm[[hmm_regime_col]].rename(columns={hmm_regime_col: "hmm_regime_id"}),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        raise ValueError("No overlapping dates between price and regime probability files.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    def _shade_regimes(ax, dates: np.ndarray, regimes: np.ndarray, title: str) -> None:
        ax.plot(dates, df["price"].to_numpy(), color="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("Price")

        # Simple color mapping by regime id (cyclic palette)
        base_colors = ["#2ca02c", "#d62728", "#7f7f7f", "#1f77b4", "#ff7f0e", "#9467bd"]
        unique_regimes = np.unique(regimes)

        boundaries = np.where(regimes[1:] != regimes[:-1])[0] + 1
        segments = np.split(np.arange(len(regimes)), boundaries)

        for seg in segments:
            r = regimes[seg[0]]
            color = base_colors[int(r) % len(base_colors)]
            start_date = dates[seg[0]]
            end_date = dates[seg[-1]]
            ax.axvspan(start_date, end_date, color=color, alpha=0.15)

    dates = df.index.to_numpy()
    _shade_regimes(
        axes[0],
        dates,
        df["gmm_regime_id"].to_numpy(dtype=int),
        title="Price with GMM regimes",
    )
    _shade_regimes(
        axes[1],
        dates,
        df["hmm_regime_id"].to_numpy(dtype=int),
        title="Price with HMM regimes",
    )

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_transition_matrices_comparison(
    gmm_trans_path: Path,
    hmm_trans_path: Path,
    output_path: Optional[Path] = None,
) -> None:
    """
    Optional diagnostic: side-by-side heatmap comparison of GMM and HMM
    transition matrices.
    """
    gmm = pd.read_csv(gmm_trans_path, index_col=0)
    hmm = pd.read_csv(hmm_trans_path, index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(
        gmm,
        ax=axes[0],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        square=True,
    )
    axes[0].set_title("GMM pseudo-transition matrix")
    axes[0].set_xlabel("j (next state)")
    axes[0].set_ylabel("i (current state)")

    sns.heatmap(
        hmm,
        ax=axes[1],
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        square=True,
    )
    axes[1].set_title("HMM learned transition matrix")
    axes[1].set_xlabel("j (next state)")
    axes[1].set_ylabel("i (current state)")

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_hmm_price(
    raw_price_path: Path,
    hmm_probs_path: Path,
    output_path: Path,
    price_col: str = "target",
    hmm_regime_col: str = "hmm_regime_id",
) -> None:
    """
    Single-model price chart with HMM regimes shaded.

    Loads a raw price panel and an HMM regime probability file
    (containing at least `hmm_regime_col`) and produces a one-panel
    chart with price and background shading by most-likely regime.
    """
    price_panel = (
        pd.read_csv(raw_price_path, index_col=0, parse_dates=True)
        .sort_index()
    )
    if price_col not in price_panel.columns:
        raise KeyError(f"Expected price column {price_col!r} in {raw_price_path}")
    prices = price_panel[price_col].rename("price")

    hmm = pd.read_csv(hmm_probs_path, index_col=0, parse_dates=True).sort_index()
    if hmm_regime_col not in hmm.columns:
        raise KeyError(f"Expected HMM regime column {hmm_regime_col!r} in {hmm_probs_path}")

    df = pd.concat(
        [
            prices,
            hmm[[hmm_regime_col]].rename(columns={hmm_regime_col: "hmm_regime_id"}),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        raise ValueError("No overlapping dates between price and HMM regime file.")

    dates = df.index.to_numpy()
    regimes = df["hmm_regime_id"].to_numpy(dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(dates, df["price"].to_numpy(), color="black", linewidth=1.0)
    ax.set_title("Gas price with HMM regimes")
    ax.set_ylabel("Price")

    base_colors = ["#2ca02c", "#d62728", "#7f7f7f", "#1f77b4", "#ff7f0e", "#9467bd"]

    boundaries = np.where(regimes[1:] != regimes[:-1])[0] + 1
    segments = np.split(np.arange(len(regimes)), boundaries)

    for seg in segments:
        r = regimes[seg[0]]
        color = base_colors[int(r) % len(base_colors)]
        start_date = dates[seg[0]]
        end_date = dates[seg[-1]]
        ax.axvspan(start_date, end_date, color=color, alpha=0.15)

    ax.set_xlabel("Date")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

