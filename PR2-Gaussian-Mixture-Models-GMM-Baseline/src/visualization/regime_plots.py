from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.gmm_engine import FEATURE_COLUMNS_14


def plot_bic_aic_silhouette(
    bics: Dict[int, float],
    aics: Dict[int, float],
    sils: Dict[int, float],
    output_path: Path,
) -> None:
    ks = sorted(bics.keys())
    bic_vals = [bics[k] for k in ks]
    aic_vals = [aics[k] for k in ks]
    sil_vals = [sils[k] for k in ks]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(ks, bic_vals, marker="o", label="BIC")
    ax1.plot(ks, aic_vals, marker="s", label="AIC")
    ax1.set_xlabel("Number of components (k)")
    ax1.set_ylabel("Information criteria")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(ks, sil_vals, marker="^", color="tab:green", label="Silhouette")
    ax2.set_ylabel("Silhouette score")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_regime_heatmap(
    profiles: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Heatmap of standardised feature means across regimes, used as a
    statistical profile of each regime.
    """
    heatmap_data = (profiles[FEATURE_COLUMNS_14] - profiles[FEATURE_COLUMNS_14].mean()) / (
        profiles[FEATURE_COLUMNS_14].std()
    )

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        center=0.0,
        yticklabels=profiles["label"],
    )
    plt.title("Standardised Feature Means by Regime")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_regimes_on_price(
    price_path: Path,
    df_regimes: pd.DataFrame,
    profiles: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot the target price series with background shading by regime,
    using a Bull/Bear/Stagnant color palette.
    """
    price_panel = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    if "target" not in price_panel.columns:
        raise KeyError("Expected 'target' column in raw price panel.")
    prices = price_panel["target"].rename("price")

    plot_df = pd.concat(
        [prices, df_regimes["regime_id"], df_regimes["regime_label"]],
        axis=1,
    ).dropna()

    color_map: Dict[int, str] = {}
    for rid, row in profiles.iterrows():
        label = row["label"]
        if "Bull" in label:
            color_map[rid] = "#2ca02c"
        elif "Bear" in label:
            color_map[rid] = "#d62728"
        else:
            color_map[rid] = "#7f7f7f"

    ids = plot_df["regime_id"].to_numpy()
    dates = plot_df.index.to_numpy()
    boundaries = np.where(ids[1:] != ids[:-1])[0] + 1
    segments = np.split(np.arange(len(ids)), boundaries)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df.index, plot_df["price"], color="black", linewidth=1.0, label="Index Price")

    for seg in segments:
        r = ids[seg[0]]
        ax.axvspan(dates[seg[0]], dates[seg[-1]], color=color_map.get(r, "#7f7f7f"), alpha=0.15)

    handles = []
    seen_labels = set()
    for rid, label in profiles["label"].items():
        if label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color_map.get(rid, "#7f7f7f"),
                lw=4,
                label=label,
            )
        )

    ax.legend(handles=handles, title="Regimes", loc="upper left")
    ax.set_ylabel("Index Level")
    ax.set_title("Price with GMM Regimes")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

